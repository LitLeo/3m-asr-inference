import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from layer.norm import MaskBatchNorm
from model.dfsmn_base_res import get_seq_mask, cFSMN_layer
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
    def __init__(self, input_dim, num_fsmn, fsmn_hidden, model_dim, num_head,
                 look_back=5, look_ahead=2, stride_left=2, stride_right=1,
                 memory_num=64, norm_type="LN", hidden_dropout=0.0, attn_dropout=0.0,
                 first_skip=False, positional_encoding=False):
        super(DFSMN_SAN_Block, self).__init__()
        self.input_dim = input_dim
        self.num_fsmn = num_fsmn
        self.model_dim = model_dim
        self.pe = None
        if positional_encoding:
            self.create_position_encoding(model_dim)
        # sub modules
        self.fsmn_layers = nn.ModuleList([])
        for i in range(num_fsmn):
            in_dim = input_dim if i == 0 else model_dim
            skip_connect = False if i == 0 and not first_skip else True
            upper_bound = None if not skip_connect else 1
            self.fsmn_layers += [
                cFSMN_layer(in_dim, fsmn_hidden, model_dim, look_back=look_back,
                            look_ahead=look_ahead, stride_left=stride_left,
                            stride_right=stride_right, dropout=hidden_dropout,
                            skip_connect=skip_connect, upper_bound=upper_bound)]
        self.attn_layer = SelfAttnMemLayer(model_dim, num_head, memory_num=memory_num,
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

    def forward(self, inputs, seq_len):
        assert inputs.dim() == 3
        batch_size, max_len, input_dim = inputs.size()
        # fsmn
        x = inputs
        for i in range(self.num_fsmn):
            x = self.fsmn_layers[i](x, seq_len)
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
                 hidden_dim=1024, memory_dim=512, look_back=4, look_ahead=1,
                 stride_left=2, stride_right=1, num_head=8, num_memory=64,
                 norm_type='LN', hidden_dropout=0.0, attn_dropout=0.0):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_dim = memory_dim
        self.num_block = num_block
        self.fsmn_each_block = fsmn_each_block
        # sub modules
        self.blocks = nn.ModuleList([])
        in_dim = input_dim
        for i in range(num_block):
            first_skip = False if i == 0 else True
            positional_encoding = True if i == 0 else False
            self.blocks += [
                    DFSMN_SAN_Block(in_dim, fsmn_each_block, hidden_dim,
                        self.model_dim, num_head, look_back=look_back,
                        look_ahead=look_ahead, stride_left=stride_left,
                        stride_right=stride_right, memory_num=num_memory,
                        norm_type=norm_type, hidden_dropout=hidden_dropout,
                        attn_dropout=attn_dropout, first_skip=first_skip,
                        positional_encoding=positional_encoding)]
            in_dim = self.model_dim
        self.out_linear = nn.Linear(self.model_dim, self.output_dim)

    def forward(self, inputs, seq_len, output_embed=False):
        assert inputs.dim() == 3
        x = inputs
        for i in range(self.num_block):
            x = self.blocks[i](x, seq_len)
        # without softmax activation
        out = self.out_linear(x)
        if output_embed:
            return out, x, seq_len
        out_softmax = F.softmax(out, dim=-1)
        return out, out_softmax, seq_len

