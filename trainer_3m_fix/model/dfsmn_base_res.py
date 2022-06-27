import torch
import math
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, input_dim, hid_dim, mem_dim, look_back=5,
                 look_ahead=5, stride_left=1, stride_right=1,
                 dropout=0.0, skip_connect=False, upper_bound=None):
        super(cFSMN_layer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hid_dim
        self.mem_dim = mem_dim
        self.look_back = look_back
        self.look_ahead = look_ahead
        self.stride_left = stride_left
        self.stride_right = stride_right
        self.upper_bound = upper_bound
        self.dropout = dropout
        self.skip_connect = skip_connect
        if skip_connect:
            assert input_dim == mem_dim
        # init memory block factor matrix
        self.left_factor = nn.Parameter(torch.zeros(look_back, mem_dim))
        self.cur_factor = nn.Parameter(torch.zeros(1, mem_dim))
        self.right_factor = nn.Parameter(torch.zeros(look_ahead, mem_dim))
        # xavier uniform for FIR filter
        nn.init.xavier_uniform_(self.left_factor, gain=0.5)
        nn.init.xavier_uniform_(self.cur_factor, gain=0.5)
        nn.init.xavier_uniform_(self.right_factor, gain=0.5)
        # sub module
        self.hid_proj = nn.Linear(input_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.mem_proj = nn.Linear(hid_dim, mem_dim, bias=False)
        # padding info for conv1d, make sure output sequence length equals input
        lctx = look_back * stride_left
        rctx = look_ahead * stride_right
        self.pad_len = max(lctx, rctx)
        # conv1d in torch only support padding the same length on both side
        # while there is different pad length for left and right sides
        # so pad with max length and crop with left and right offset
        self.left_offset = max(0, self.pad_len - lctx)
        self.right_offset = min(0, rctx - self.pad_len)

    def forward(self, inputs, seq_len):
        """
        Args:
            - inputs: [batch_size, max_steps, dim]
            - seq_len: [batch_size]
        Returns:
            - mem: output memory block
        """
        assert inputs.dim() == 3
        batch_size, max_steps, input_dim = inputs.size()
        seq_mask = get_seq_mask(seq_len, max_steps).float()
        # projection, [batch_size, max_steps, mem_dim]
        hid = F.relu(self.hid_proj(inputs))
        # restrict hidden under the upper bound
        if self.upper_bound is not None:
            hid = torch.min(hid, hid.new(hid.size()).fill_(self.upper_bound))
        hid = self.dropout(hid)
        # linear activation and mask the padding frames
        p = self.mem_proj(hid)
        if self.skip_connect:
            p = p + inputs
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
        ###if self.skip_connect:
        ###    mem = inputs + mem
        return mem


class Net(nn.Module):
    """
    Deep FSMN with multiple cFSMN layers
    """
    def __init__(self, input_dim, output_dim, fsmn_layers=30, hidden_dim=1024,
                 memory_dim=512, look_back=4, look_ahead=1, stride_left=2,
                 stride_right=1, dropout=0.0):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fsmn_layers = nn.ModuleList([])
        for i in range(fsmn_layers):
            in_dim = input_dim if i == 0 else memory_dim
            skip_connect = False if i == 0 else True
            upper_bound = None if i == 0 else 1
            self.fsmn_layers += [
                cFSMN_layer(in_dim, hidden_dim, memory_dim,
                            look_back=look_back, look_ahead=look_ahead,
                            stride_left=stride_left, stride_right=stride_right,
                            dropout=dropout, skip_connect=skip_connect,
                            upper_bound=upper_bound)]
        self.out_linear = nn.Linear(memory_dim, output_dim)

    def forward(self, inputs, seq_len):
        """
        Args:
            - inputs: [batch_size, max_steps, input_dim]
            - seq_len: [batch_size]
        Returns:
            - out: projected output before softmax
        """
        assert inputs.dim() == 3
        fsmn_input = inputs
        for i in range(len(self.fsmn_layers)):
            fsmn_input = self.fsmn_layers[i](fsmn_input, seq_len)
        # without softmax activation
        out = self.out_linear(fsmn_input)
        return out, seq_len

