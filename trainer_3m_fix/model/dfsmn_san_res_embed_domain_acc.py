import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from layer.norm import MaskBatchNorm
from model.dfsmn_base_res import get_seq_mask, cFSMN_layer


class MultiHeadAttnMemLayer(nn.Module):
    """
    Multi-head attention layer with or without consistent memory
    Args:
        - model_dim: dimension of query, key and value
        - head_num: number of attention heads
        - memory_num: number of memory vectors each head
    """
    def __init__(self, model_dim, head_num, memory_num=0, dropout=0.0):
        assert model_dim % head_num == 0
        super(MultiHeadAttnMemLayer, self).__init__()
        self.model_dim = model_dim
        self.head_num = head_num
        self.memory_num = memory_num
        self.dim_each_head = model_dim // head_num
        if memory_num > 0:
            self.key_memory = nn.Parameter(
                    torch.zeros(head_num, memory_num, self.dim_each_head))
            nn.init.xavier_uniform_(self.key_memory)
            self.value_memory = nn.Parameter(
                    torch.zeros(head_num, memory_num, self.dim_each_head))
            nn.init.xavier_uniform_(self.value_memory)

        self.linear_query = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_key = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_value = nn.Linear(model_dim, model_dim, bias=False)
        self.linear_out = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, key, value, query, mask=None):
        """
        Compute the context vector and the attention vectors
        Args:
            - key (FloatTensor): shape is [batch, key_len, model_dim]
            - value (FloatTensor): shape is [batch, key_len, model_dim]
            - query (FloatTensor): shape is [batch, query_len, model_dim]
            - mask (BooleanTensor): mask area which shouldn't be attended, if None,
                                    shape should be [batch, query_len, key_len]
        Returns:
            - out: output context vectors, [batch, seq_len, model_dim]
            - attn_vec: attention vectors on keys, [batch, head_num, seq_len, seq_len]
            - attn_mem: attention vectors on memorys, None or [batch, head_num, seq_len, memory_num]
        """
        b_k, l_k, d_k = key.size()
        b_v, l_v, d_v = value.size()
        b_q, l_q, d_q = query.size()
        assert b_k == b_v and b_k == b_q
        assert l_k == l_v
        if mask is not None:
            b_m, l_m, d_m = mask.size()
            assert b_m == b_q and l_m == l_q and d_m == l_k
        head_num = self.head_num
        memory_num = self.memory_num
        dim_each_head = self.dim_each_head
        # fold each head into batch dimension
        def shape_projection(x):
            b, l, d = x.size()
            return x.view(b, l, head_num, dim_each_head) \
                    .transpose(1, 2).contiguous() \
                    .view(b * head_num, l, dim_each_head)
        # unfold each head into last dimension
        def unshape_projection(x):
            bh, l, d = x.size()
            b = bh // head_num
            return x.view(b, head_num, l, dim_each_head) \
                    .transpose(1, 2).contiguous() \
                    .view(b, l, head_num * dim_each_head)

        key_up = shape_projection(self.linear_key(key))
        value_up = shape_projection(self.linear_value(value))
        if memory_num > 0:
            expand_key_mem = self.key_memory.repeat(b_k, 1, 1)
            key_up = torch.cat([key_up, expand_key_mem], dim=1)
            expand_value_mem = self.value_memory.repeat(b_v, 1, 1)
            value_up = torch.cat([value_up, expand_value_mem], dim=1)
        query_up = shape_projection(self.linear_query(query))
        # [batch * head_num, query_len, key_len]
        score = torch.bmm(query_up, key_up.transpose(1, 2))
        score = score / math.sqrt(self.dim_each_head)
        bh, ql, kl = score.size()
        b = bh // self.head_num
        if mask is not None:
            score = score.view(b, self.head_num, ql, kl)
            if memory_num > 0:
                pad_mem_mask = mask.new_zeros((b, ql, memory_num))
                mask = torch.cat([mask, pad_mem_mask], dim=2)
            # expand along attention head dimension
            mask = mask.unsqueeze(1).expand_as(score)
            score = score.masked_fill(mask, -float('inf')) \
                         .view(bh, ql, kl)
        attn = self.sm(score)
        attn = self.dropout(attn)
        # [batch, l, head_num * dim_each_head]
        out = self.linear_out(unshape_projection(torch.bmm(attn, value_up)))
        attn_vec = attn.view(b, self.head_num, ql, kl)
        attn_mem = None
        if memory_num > 0:
            attn_mem = attn_vec[:, :, :, -memory_num:]
            attn_vec = attn_vec[:, :, :, 0:-memory_num]
        return out, attn_vec, attn_mem


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
        self.hid_proj_att = nn.Linear(model_dim, model_dim)
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
        #####attn_out, attn_vec, attn_mem = self.attn_layer(x, x, x, mask=attn_mask)
        hid = F.relu(self.hid_proj_att(x))
        attn_out, attn_vec, attn_mem = self.attn_layer(hid, hid, hid, mask=attn_mask)
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
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.output_dim_domain = cfg.output_dim_domain
        self.output_dim_acc = cfg.output_dim_acc
        self.model_dim = cfg.memory_dim
        self.num_block = cfg.num_block
        self.fsmn_each_block = cfg.fsmn_each_block
        # sub modules
        self.blocks = nn.ModuleList([])
        in_dim = cfg.input_dim
        for i in range(cfg.num_block):
            first_skip = False if i == 0 else True
            positional_encoding = True if i == 0 else False
            self.blocks += [
                    DFSMN_SAN_Block(in_dim, cfg.fsmn_each_block, cfg.hidden_dim,
                        self.model_dim, cfg.num_head, look_back=cfg.look_back,
                        look_ahead=cfg.look_ahead, stride_left=cfg.stride_left,
                        stride_right=cfg.stride_right, memory_num=cfg.num_memory,
                        norm_type=cfg.norm_type, hidden_dropout=cfg.hidden_dropout,
                        attn_dropout=cfg.attn_dropout, first_skip=first_skip,
                        positional_encoding=positional_encoding)]
            in_dim = self.model_dim
        self.out_linear = nn.Linear(self.model_dim, self.output_dim)
        self.out_linear_domain = nn.Linear(self.model_dim, self.output_dim_domain)
        self.out_linear_accent = nn.Linear(self.model_dim, self.output_dim_acc)
        self.out_linear_domain_embed = nn.Linear(self.model_dim, self.model_dim)
        self.out_linear_accent_embed = nn.Linear(self.model_dim, self.model_dim)
       ##### self.out_linear_acc = nn.Linear(self.model_dim, self.output_dim_acc)

    def forward(self, inputs, seq_len, output_embed=False):
        assert inputs.dim() == 3
        x = inputs
        for i in range(self.num_block):
            x = self.blocks[i](x, seq_len)
        # without softmax activation
        x_domain = self.out_linear_domain_embed(x)
        x_acc = self.out_linear_accent_embed(x)
        x_pool_domain = torch.mean(x_domain, 1, True)
        x_pool_acc = torch.mean(x_acc, 1, True)
        out_pool = self.out_linear_domain(x_pool_domain)
        out_pool_acc = self.out_linear_accent(x_pool_acc)
        x_pool_domain_exp = x_pool_domain.expand(-1, x.shape[1], -1)
        x_pool_acc_exp = x_pool_acc.expand(-1, x.shape[1], -1)
        ####x_pool_exp_acc = x_pool.expand(-1, x.shape[1], -1)
        out = self.out_linear(x)
        x_cat = torch.cat((x, x_pool_domain_exp), dim=-1)
        x_cat_2 = torch.cat((x_cat, x_pool_acc_exp), dim=-1)
        if output_embed:
            ###return out, x, seq_len
            return out, out_pool, out_pool_acc, x_cat_2, seq_len
        out_softmax = F.softmax(out, dim=-1)
        ####return out, out_softmax, seq_len
        return out, out_pool, out_pool_acc, out_softmax, seq_len


def register(parser):
    """
    add argument for model definition
    """
    parser.add_argument("--output_dim", type=int, default=1434,
            help="output dimension of the model")
    parser.add_argument("--output_dim_domain", type=int, default=6,
            help="output domain dimension of the model")
    parser.add_argument("--output_dim_acc", type=int, default=8,
            help="output acc dimension of the model")
    parser.add_argument("--hidden_dim", type=int, default=1024,
            help="num units in the hidden layer of model")
    parser.add_argument("--memory_dim", type=int, default=512,
            help="num units of the memory block")
    parser.add_argument("--hidden_dim", type=int, default=1024,
            help="num units in the hidden layer of model")
    parser.add_argument("--memory_dim", type=int, default=512,
            help="num units of the memory block")
    parser.add_argument("--look_back", type=int, default=5,
            help="window size of the history")
    parser.add_argument("--look_ahead", type=int, default=5,
            help="window size of the future")
    parser.add_argument("--stride_left", type=int, default=1,
            help="stride for the history context")
    parser.add_argument("--stride_right", type=int, default=1,
            help="stride for the future context")
    parser.add_argument("--num_head", type=int, default=8,
            help="number of heads in self-attn layers")
    parser.add_argument("--num_memory", type=int, default=0,
            help="number of memory vectors in self-attn layers")
    parser.add_argument("--norm_type", type=str, default="BN",
            choices=["LN", "BN"], help="type of normalization to be used")
    parser.add_argument("--num_block", type=int, default=3,
            help="number of block which consists of DFSMN and SAN")
    parser.add_argument("--fsmn_each_block", type=int, default=10,
            help="number of FSMN layers in one block")
    parser.add_argument("--hidden_dropout", type=float, default=0.0,
            help="dropout probability for hidden layer in DFSMN")
    parser.add_argument("--attn_dropout", type=float, default=0.0,
            help="dropout probability for attention probability in self-attn")

def log_model_info(logging, cfg):
    """
    log model information
    """
    logging.info('input  dim: {},\toutput dim: {}'.format(
                 cfg.input_dim, cfg.output_dim))
    logging.info('block num: {},\tfsmn layers each block: {}'.format(
                 cfg.num_block, cfg.fsmn_each_block))
    logging.info('hidden dim: {},\tmemory dim: {}'.format(
                 cfg.hidden_dim, cfg.memory_dim))
    logging.info('history order: {},\tfuture order: {}'.format(
                 cfg.look_back, cfg.look_ahead))
    logging.info('stride for history: {},\tstride for future: {}'.format(
                 cfg.stride_left, cfg.stride_right))
    logging.info('attn memory num: {}, head num: {}'.format(
                 cfg.num_memory, cfg.num_head))

