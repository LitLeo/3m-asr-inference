import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from loss.balance_loss import SparseL1Loss, BalanceImportanceLoss

from fmoe.layers import FMoELinear
from fmoe.functions import moe_prepare_forward
from fmoe.functions import MOEScatter, MOEGather


def get_seq_mask(seq_len, max_len=None):
    """
    compute mask for valid frames from sequence length
    """
    batch_size = seq_len.size(0)
    max_len = max_len or seq_len.max().item()
    mask = (torch.arange(0, max_len).
            type_as(seq_len).
            repeat(batch_size, 1).
            lt(seq_len.unsqueeze(1)))
    return mask


def mark_module_parallel_comm(m, dp_comm):
    for p in m.parameters():
        setattr(p, "dp_comm", dp_comm)


def _fmoe_general_global_forward(inp, gate, expert_fn, num_expert, world_size,
                                 capacity=-1, comm=None):
    r"""
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    ) = moe_prepare_forward(gate, num_expert, world_size, comm=comm)
    x = MOEScatter.apply(
        inp, pos,
        local_expert_count, global_expert_count, fwd_batch_size, world_size
    )
    x = expert_fn(x, fwd_expert_count, capacity=capacity)
    x = MOEGather.apply(
        x, pos, local_expert_count, global_expert_count, inp.shape[0], world_size
    )
    return x


class Expert(nn.Module):
    def __init__(self, num_experts, d_model, d_hidden, rank=0,
                 dropout=0.0, upper_bound=None):
        super(Expert, self).__init__()
        self.hid_proj = FMoELinear(num_experts, d_model, d_hidden, bias=True, rank=rank)
        self.mem_proj = FMoELinear(num_experts, d_hidden, d_model, bias=False, rank=rank)
        self.dropout = nn.Dropout(dropout)
        self.upper_bound = upper_bound

    def forward(self, inp, fwd_expert_count, capacity=-1):
        h = self.hid_proj(inp, fwd_expert_count, capacity=capacity)
        h = F.relu(h)
        if self.upper_bound is not None:
            h = h.clamp(max=self.upper_bound)
        h = self.dropout(h)
        m = self.mem_proj(h, fwd_expert_count, capacity=capacity)
        return m


class cFSMN_layer(nn.Module):
    """
    Implementation of compact FSMN layer
    Args:
        - input_dim: dimension of input
        - hid_dim: dimension of hidden affine
        - mem_dim: dimension of memory projection
        - look_back: order of left context in FIR filter
        - look_ahead: order of right context in FIR filter
        - stride_left: stride for left context
        - stride_right: stride for right context
        - skip_connect: whether to use skip connection
        - upper_bound: upper bound for hidden affine result
    """
    def __init__(self, input_dim, embed_dim, hid_dim=1024, mem_dim=512,
                 look_back=5, look_ahead=5, stride_left=1, stride_right=1,
                 num_experts=4, rank=0, world_size=1, capacity_factor=1,
                 dropout=0.0, skip_connect=False, upper_bound=None,
                 router_regularization="l1_plus_importance", ln_before_router=False,
                 detach_router_input=False, router_with_bias=False,
                 non_expert_dropout=0.0, rand_init_router=False, comm=None):
        super(cFSMN_layer, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hid_dim
        self.mem_dim = mem_dim
        self.look_back = look_back
        self.look_ahead = look_ahead
        self.stride_left = stride_left
        self.stride_right = stride_right
        self.upper_bound = upper_bound
        self.skip_connect = skip_connect
        self.num_experts = num_experts
        self.rank = rank
        self.world_size = world_size
        self.comm = comm
        self.capacity_factor = capacity_factor
        self.ln_before_router = ln_before_router
        self.detach_router_input = detach_router_input
        self.router_regularization = router_regularization
        self.router_with_bias = router_with_bias
        assert router_regularization in ["l1_plus_importance"]
        if skip_connect:
            assert input_dim == mem_dim
            self.experts = Expert(num_experts, mem_dim, hid_dim,
                    rank=rank, dropout=dropout, upper_bound=upper_bound)
        else:
            self.hid_proj = nn.Linear(input_dim, hid_dim)
            self.mem_proj = nn.Linear(hid_dim, mem_dim, bias=False)
            self.non_expert_dropout = nn.Dropout(non_expert_dropout)
        # init memory block factor matrix
        self.left_factor = nn.Parameter(torch.zeros(look_back, mem_dim))
        self.cur_factor = nn.Parameter(torch.zeros(1, mem_dim))
        self.right_factor = nn.Parameter(torch.zeros(look_ahead, mem_dim))
        # rooter
        self.rooter_weights = nn.Parameter(torch.zeros(embed_dim + input_dim, num_experts * world_size))
        if self.router_with_bias:
            self.rooter_bias = nn.Parameter(torch.zeros(num_experts * world_size))
        if self.ln_before_router:
            self.ln_for_router = nn.LayerNorm(embed_dim + input_dim)
        self.sparseLoss = SparseL1Loss(self.world_size)
        self.balanceLoss = BalanceImportanceLoss(self.world_size)

        # xavier uniform for FIR filter
        nn.init.xavier_uniform_(self.left_factor, gain=0.5)
        nn.init.xavier_uniform_(self.cur_factor, gain=0.5)
        nn.init.xavier_uniform_(self.right_factor, gain=0.5)
        if rand_init_router:
            nn.init.xavier_uniform_(self.rooter_weights, gain=0.5)

        # padding info for conv1d, make sure output sequence length equals input
        lctx = look_back * stride_left
        rctx = look_ahead * stride_right
        self.pad_len = max(lctx, rctx)
        # conv1d in torch only support padding the same length on both side
        # while there is different pad length for left and right sides
        # so pad with max length and crop with left and right offset
        self.left_offset = max(0, self.pad_len - lctx)
        self.right_offset = min(0, rctx - self.pad_len)

        if self.skip_connect:
            mark_module_parallel_comm(self.experts, "mp")
            setattr(self.rooter_weights, "dp_comm", "dp_mean")
            if self.router_with_bias:
                setattr(self.rooter_bias, "dp_comm", "dp_mean")

    def gate(self, inputs):
        """
        inputs shape: [num_frames, d_model]

        """
        if self.detach_router_input:
            inputs = inputs.clone().detach()
        if self.ln_before_router:
            inputs = self.ln_for_router(inputs)

        router_logits = torch.einsum('ij,jk->ik', [inputs, self.rooter_weights])
        if self.router_with_bias:
            router_logits = router_logits + self.rooter_bias
        router_probs = F.softmax(router_logits, dim=-1)
        gate_value, gate_idx = router_probs.max(dim=-1)
        all_samples = router_probs.size(0)

        # router regularization
        if self.router_regularization == "l1_plus_importance":
            l1_loss, l1_loss_item, n_samples = self.sparseLoss(router_probs, group=self.comm)
            all_samples = n_samples
            importance_loss, importance_loss_item = self.balanceLoss(router_probs, group=self.comm)
            aux_loss = ((l1_loss, l1_loss_item), (importance_loss, importance_loss_item))
        else:
            raise NotImplementedError("Not supported router_regularization type: {}".format(
                                      self.router_regularization))
        return gate_idx, gate_value, aux_loss, all_samples

    def forward(self, inputs, embed, seq_len, is_training, stick_router=False,
                keep_expert_output=False):
        """
        Args:
            - inputs: [batch_size, max_steps, dim]
            - seq_len: [batch_size]
        Returns:
            - mem: output memory block
        """
        assert inputs.dim() == 3
        xx = inputs
        if self.skip_connect:
            batch_size, max_steps, input_dim = inputs.size()
            inputs = inputs.view(-1, input_dim)
            embed_dim = embed.size(-1)
            embed = embed.view(-1, embed_dim)
            embed_inputs = torch.cat([embed, inputs], dim=-1)
            gate_idx, gate_value, aux_loss, all_samples = self.gate(embed_inputs)
            all_experts = self.num_experts * self.world_size
            capacity = int(self.capacity_factor * all_samples / all_experts)
            expert_outputs = _fmoe_general_global_forward(
                    inputs, gate_idx, self.experts, self.num_experts,
                    self.world_size, capacity=capacity, comm=self.comm)
            if not keep_expert_output:
                expert_outputs = expert_outputs * gate_value.unsqueeze(1)
            p = expert_outputs.view(batch_size, max_steps, self.mem_dim)
            seq_mask = get_seq_mask(seq_len, max_steps).float()
            # residual connect in feed-forward
            p = p + xx
            # mask the padding frames before convolution 
            p = p * seq_mask.unsqueeze(2)
            # pad factor matrix with zero according to strides
            lctx = self.look_back * self.stride_left
            rctx = self.look_ahead * self.stride_right
            f_mat = p.data.new(lctx + 1 + rctx, self.mem_dim).zero_()
            f_mat[0:lctx:self.stride_left] = self.left_factor
            f_mat[lctx:lctx+1] = self.cur_factor
            f_mat[lctx+self.stride_right::self.stride_right] = self.right_factor
            """compute memory with groups conv1d on time dimension"""
            conv_w = f_mat.transpose(0, 1).contiguous().unsqueeze(1)
            # conv1d input shape is [batch, in_channel, W]
            # groups = in_channel, each channel conv individually, no bias
            conv = F.conv1d(p.transpose(1, 2), conv_w, padding=self.pad_len,
                            groups=self.mem_dim)
            conv_len = conv.size(2)
            mem = conv[:, :, self.left_offset: conv_len + self.right_offset]
            # back to shape [batch, time_step, mem_dim] and add p
            mem = mem.transpose(1, 2).contiguous() + p
        else:
            batch_size, max_steps, input_dim = inputs.size()
            seq_mask = get_seq_mask(seq_len, max_steps).float()
            # projection, [batch_size, max_steps, mem_dim]
            hid = F.relu(self.hid_proj(inputs))
            # restrict hidden under the upper bound
            if self.upper_bound is not None:
                hid = hid.clamp(max=self.upper_bound)
            hid = self.non_expert_dropout(hid)
            # linear activation and mask the padding frames
            p = self.mem_proj(hid)
            p = p * seq_mask.unsqueeze(2)
            # pad factor matrix with zero according to strides
            lctx = self.look_back * self.stride_left
            rctx = self.look_ahead * self.stride_right
            f_mat = p.data.new(lctx + 1 + rctx, self.mem_dim).zero_()
            f_mat[0:lctx:self.stride_left] = self.left_factor
            f_mat[lctx:lctx+1] = self.cur_factor
            f_mat[lctx+self.stride_right::self.stride_right] = self.right_factor
            """compute memory with groups conv1d on time dimension"""
            conv_w = f_mat.transpose(0, 1).contiguous().unsqueeze(1)
            # conv1d input shape is [batch, in_channel, W]
            # groups = in_channel, each channel conv individually, no bias
            conv = F.conv1d(p.transpose(1, 2), conv_w, padding=self.pad_len,
                            groups=self.mem_dim)
            conv_len = conv.size(2)
            mem = conv[:, :, self.left_offset: conv_len + self.right_offset]
            # back to shape [batch, time_step, mem_dim] and add p
            mem = mem.transpose(1, 2).contiguous() + p
        # if self.skip_connect:
        #     mem = xx + mem
        if self.skip_connect:
            return mem, aux_loss
        else:
            return mem, 0

