import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from collections import OrderedDict
from layer.norm import MaskBatchNorm
from model.dfsmn_base_fmoe_localComm_catEmbed import get_seq_mask, cFSMN_layer
from model.dfsmn_san_res_embed import Net as dfsmnsan_embed
from layer.attention import MultiHeadAttnMemLayer


class SelfAttnMemLayer(nn.Module):
    """
    Composed of a self-attn layer(w/o memery) and a normalization(BN/LN) layer
    out = Norm(input + Self-Attn-Memory(input))
    Args:
        - model_dim: dimension of self attention
        - head_num: number of attention heads
        - memory_num: number of consistent memory vectors
        - norm_type: LN or BN
    """
    def __init__(self, model_dim, head_num, memory_num=0,
                 norm_type="LN", dropout=0.0):
        super(SelfAttnMemLayer, self).__init__()
        assert model_dim % head_num == 0
        self.model_dim = model_dim
        self.head_num = head_num
        self.memory_num = memory_num
        self.norm_type = norm_type
        assert norm_type in ["LN", "BN"]

        self.attn_layer = MultiHeadAttnMemLayer(model_dim, head_num,
                memory_num, dropout=dropout)
        if norm_type == "LN":
            self.ln_layer = nn.LayerNorm(model_dim)
        else:
            self.bn_layer = MaskBatchNorm([1, model_dim])

    def forward(self, x, seq_len):
        batch, max_len, dim = x.size()
        assert dim == self.model_dim
        # indicate valid frame in each sentence
        seq_mask = get_seq_mask(seq_len, max_len)
        # indicate padding position in each sentence
        attn_mask = ~seq_mask.unsqueeze(1).expand(batch, max_len, max_len)
        attn_out, attn_vec, attn_mem = self.attn_layer(x, x, x, mask=attn_mask)
        x = x + attn_out
        if self.norm_type == "LN":
            x = self.ln_layer(x)
        else:
            # transform into stacked frames
            flat_mask = seq_mask.view(-1)
            x = x.view(-1, self.model_dim)
            # only batch norm with valid frames in batch
            x = self.bn_layer(x, flat_mask, [0])
            x = x.view(batch, -1, self.model_dim)
        return x


class DFSMN_SAN_Block(nn.Module):
    def __init__(self, input_dim, embed_dim, num_fsmn, hidden_dim=1024, memory_dim=512,
                 num_head=8, look_back=5, look_ahead=2, stride_left=2, stride_right=1,
                 num_memory=64, num_experts=4, rank=0, world_size=1,
                 capacity_factor=1, norm_type="LN", hidden_dropout=0.0,
                 attn_dropout=0.0, first_skip=False, positional_encoding=False,
                 router_regularization="l1_plus_importance", ln_before_router=False,
                 detach_router_input=False, router_with_bias=False,
                 non_expert_dropout=0.0, rand_init_router=False, comm=None):
        super(DFSMN_SAN_Block, self).__init__()
        self.input_dim = input_dim
        self.num_fsmn = num_fsmn
        self.model_dim = memory_dim
        self.pe = None
        if positional_encoding:
            self.create_position_encoding(self.model_dim)
        # sub modules
        self.fsmn_layers = nn.ModuleList([])
        for i in range(num_fsmn):
            in_dim = input_dim if i == 0 else memory_dim
            skip_connect = False if i == 0 and not first_skip else True
            upper_bound = None if not skip_connect else 1
            self.fsmn_layers += [
                cFSMN_layer(in_dim, embed_dim, hidden_dim, memory_dim, look_back=look_back,
                            look_ahead=look_ahead, stride_left=stride_left, stride_right=stride_right,
                            num_experts=num_experts, rank=rank, world_size=world_size,
                            capacity_factor=capacity_factor, dropout=hidden_dropout,
                            skip_connect=skip_connect, upper_bound=upper_bound,
                            router_regularization=router_regularization, ln_before_router=ln_before_router,
                            detach_router_input=detach_router_input, router_with_bias=router_with_bias,
                            non_expert_dropout=non_expert_dropout, rand_init_router=rand_init_router,
                            comm=comm)]
        self.attn_layer = SelfAttnMemLayer(self.model_dim, num_head, memory_num=num_memory,
                                           norm_type=norm_type, dropout=attn_dropout)

    def create_position_encoding(self, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        self.pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        self.pe[:, 0::2] = torch.sin(position.float() * div_term)
        self.pe[:, 1::2] = torch.cos(position.float() * div_term)
        # [B, T, D]
        self.pe = self.pe.unsqueeze(0)

    def forward(self, inputs, embed, seq_len, aux_loss, is_training,
                stick_router=False, keep_expert_output=False):
        assert inputs.dim() == 3
        batch_size, max_len, input_dim = inputs.size()
        # fsmn
        x = inputs
        for i in range(self.num_fsmn):
            x, loss = self.fsmn_layers[i](x, embed, seq_len, is_training, stick_router=stick_router,
                                          keep_expert_output=keep_expert_output)
            if loss != 0:
                aux_loss.append(loss)
            ##aux_loss.append(loss)
        # positional encoding
        if self.pe is not None:
            if max_len > self.pe.size(1):
                self.create_position_encoding(self.pe.size(2), max_len)
            self.pe = self.pe.to(x.device)
            x = x + self.pe[:, 0:max_len, :]
        # self attention
        x = self.attn_layer(x, seq_len)
        return x


class Net(nn.Module):
    """
    DFSMN + SAN(w/o consistent memory), SAN is placed after N layer FSMN
    """
    def __init__(self, input_dim, output_dim, num_block=3, fsmn_each_block=10,
                 embed_before_output=True, embed_conf=None, fsmn_conf=None,
                 san_conf=None, moe_conf=None):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_before_output = embed_before_output
        # embed_conf
        self.embed_conf = {
            'num_block': 3,
            'fsmn_each_block': 10,
            'hidden_dim': 1024,
            'memory_dim': 512,
            'look_back': 4,
            'look_ahead': 1,
            'stride_left': 2,
            'stride_right': 1,
            'hidden_dropout': 0.0,
            'num_head': 8,
            'num_memory': 64,
            'norm_type': 'LN',
            'attn_dropout': 0.0
        }
        if embed_conf is not None and isinstance(embed_conf, dict):
            self.embed_conf.update(embed_conf)
        self.embed = dfsmnsan_embed(input_dim, output_dim, **self.embed_conf)
        # fsmn_conf
        self.fsmn_conf = {
            'hidden_dim': 1024,
            'memory_dim': 512,
            'look_back': 4,
            'look_ahead': 1,
            'stride_left': 2,
            'stride_right': 1,
            'hidden_dropout': 0.0
        }
        if fsmn_conf is not None and isinstance(fsmn_conf, dict):
            self.fsmn_conf.update(fsmn_conf)
        # self-attention conf
        self.san_conf = {
            'num_head': 8,
            'num_memory': 64,
            'norm_type': 'LN',
            'attn_dropout': 0.0
        }
        if san_conf is not None and isinstance(san_conf, dict):
            self.san_conf.update(san_conf)
        # moe conf
        self.moe_conf = {
            'rank': 0,
            'world_size': 1,
            'comm': None,
            'num_experts': 4,
            'capacity_factor': -1,
            'keep_expert_output': False,
            'router_regularization': 'l1_plus_importance',
            'ln_before_router': False,
            'detach_router_input': False,
            'router_with_bias': False,
            'non_expert_dropout': 0.0,
            'rand_init_router': False
        }
        if moe_conf is not None and isinstance(moe_conf, dict):
            self.moe_conf.update(moe_conf)
        self.keep_expert_output = self.moe_conf['keep_expert_output']
        # whole_conf
        block_conf = {}
        block_conf.update(self.fsmn_conf)
        block_conf.update(self.san_conf)
        block_conf.update(self.moe_conf)

        self.model_dim = self.fsmn_conf['memory_dim']
        self.blocks_sw = nn.ModuleList([])
        embed_dim = self.embed_conf['memory_dim'] if self.embed_before_output else output_dim
        in_dim = input_dim
        for i in range(num_block):
            first_skip = False if i == 0 else True
            positional_encoding = True if i == 0 else False
            block_conf['first_skip'] = first_skip
            block_conf['positional_encoding'] = positional_encoding
            self.blocks_sw += [
                DFSMN_SAN_Block(in_dim, embed_dim, fsmn_each_block, **block_conf)
            ]
            in_dim = self.model_dim

        self.out_linear_sw = nn.Linear(self.model_dim, self.output_dim)

    def forward(self, inputs, seq_len, is_training=False, stick_router=False):
        assert inputs.dim() == 3
        output_embed = True if self.embed_before_output else False
        x = inputs
        if stick_router:
            with torch.no_grad():
                embed_out, embed, x_len = self.embed(x, seq_len, output_embed=output_embed)
        else:
            embed_out, embed, x_len = self.embed(x, seq_len, output_embed=output_embed)
        embed = embed.detach()
        aux_loss=[]
        for i in range(self.num_block):
            x = self.blocks_sw[i](x, embed, seq_len, aux_loss, is_training, stick_router=stick_router,
                                  keep_expert_output=self.keep_expert_output)
        # without softmax activation
        out = self.out_linear_sw(x)
        return out, embed_out, seq_len, aux_loss

    def state_dict_comm(self):
        local_state_dict = self.state_dict()
        if self.world_size <= 1:
            return local_state_dict
        else:
            new_state_dict = OrderedDict()
            all_experts_num = self.world_size * self.num_experts
            for k, v in local_state_dict.items():
                if "experts" not in k:
                    new_state_dict[k] = v
                else:
                    new_size = list(v.size())
                    new_size[0] = all_experts_num
                    experts_weight = v.data.new_zeros(*new_size)
                    experts_weight[self.rank * self.num_experts: (self.rank + 1) * self.num_experts] = v
                    dist.all_reduce(experts_weight, group=self.comm, async_op=False)
                    new_state_dict[k] = experts_weight
            return new_state_dict

    def load_state_dict_comm(self, whole_model_state):
        if self.world_size <= 1:
            return self.load_state_dict(whole_model_state)
        else:
            new_state_dict = OrderedDict()
            for k, v in whole_model_state.items():
                if "experts" not in k:
                    new_state_dict[k] = v
                else:
                    assert v.size(0) == self.num_experts * self.world_size
                    new_state_dict[k] = v[self.rank * self.num_experts: (self.rank + 1) * self.num_experts]
            return self.load_state_dict(new_state_dict)

