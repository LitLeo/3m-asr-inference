from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from layer.transformer import ConformerEncoderLayer
from layer.positionwise_feed_forward import PositionwiseFeedForward
from layer.subsampling import LinearNoSubsampling
from layer.subsampling import Conv2dSubsampling4
from layer.subsampling import Conv2dSubsampling6
from layer.subsampling import Conv2dSubsampling8
from layer.positional_encoding import PositionalEncoding
from layer.positional_encoding import RelPositionalEncoding
from layer.positional_encoding import NoPositionalEncoding
from layer.attention import MultiHeadedAttention
from layer.attention import RelPositionMultiHeadedAttention
from layer.convolution import ConvolutionModule
from utils.common import get_activation
from utils.mask import make_pad_mask
from utils.mask import add_optional_chunk_mask

import tensorrt as trt
import trt_helper

class Net(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        attention_heads: int = 4,
        attention_dim: int = 256,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        conv_subsample_in_ch: int = 1,
        output_dim_domain: int = 6,
        output_dim_acc: int = 8
    ):
        """
        Conformer from Wenet Implementation
        """
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim_domain = output_dim_domain
        self.output_dim_acc = output_dim_acc
        # subsampling
        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
            subsampling_args = (input_dim, attention_dim, dropout)
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
            assert input_dim % conv_subsample_in_ch == 0
            subsampling_args = (input_dim // conv_subsample_in_ch, attention_dim, conv_subsample_in_ch)
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
            assert input_dim % conv_subsample_in_ch == 0
            subsampling_args = (input_dim // conv_subsample_in_ch, attention_dim, conv_subsample_in_ch)
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
            assert input_dim % conv_subsample_in_ch == 0
            subsampling_args = (input_dim // conv_subsample_in_ch, attention_dim, conv_subsample_in_ch)
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.subsampling = subsampling_class(*subsampling_args)
        # positional embeding
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)
        self.pos_enc = pos_enc_class(attention_dim, positional_dropout_rate)
        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-12)
        ###self.after_norm_domain = torch.nn.LayerNorm(4, eps=1e-12)
        ###self.after_norm_acc = torch.nn.LayerNorm(4, eps=1e-12)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        # attention layer
        activation = get_activation(activation_type)
        if pos_enc_layer_type == "no_pos":
            selfattn_layer = MultiHeadedAttention
        else:
            selfattn_layer = RelPositionMultiHeadedAttention
        san_layer_args = (
            attention_heads,
            attention_dim,
            attention_dropout_rate,
        )
        # feed-forward module in attention
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            attention_dim,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module in attention
        convolution_layer = ConvolutionModule
        convolution_layer_args = (
            attention_dim,
            cnn_module_kernel,
            activation,
            cnn_module_norm,
            causal,
        )
        # encoder blocks
        self.blocks = torch.nn.ModuleList([
            ConformerEncoderLayer(
                attention_dim,
                selfattn_layer(*san_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(num_blocks)
        ])
        # output layer
        self.out_linear = nn.Linear(attention_dim, self.output_dim)
        ###self.out_linear_domain = nn.Linear(4, self.output_dim_domain)
        ###self.out_linear_accent = nn.Linear(4, self.output_dim_acc)
        ###self.out_linear_domain_embed = nn.Linear(attention_dim, 4)
        ###self.out_linear_accent_embed = nn.Linear(attention_dim, 4)

    def forward(
        self,
        network_helper,
        xs : trt.ITensor,
        xs_len: trt.ITensor,
        output_embed: bool = False
    ) :
        # import pdb
        # pdb.set_trace()
        xs, xs_len = self.subsampling(network_helper, xs, xs_len)
        xs, pos_emb = self.pos_enc(network_helper, xs)

        # xs = network_helper.addDumpTensor(xs, "before transformer")
        # decoded_frame_num = network_helper.addDumpTensor(decoded_frame_num, "decoded_frame_num")
        # xs_len = network_helper.addDumpTensor(xs_len, "xs_len")

        for layer in self.blocks:
            xs, xs_len = layer(network_helper, xs, xs_len, pos_emb)

        # xs = network_helper.addDumpTensor(xs, "transformer out")

        if self.normalize_before:
            # xs = self.after_norm(xs)
            xs = network_helper.addLayerNorm(self.after_norm, xs)

        # out = self.out_linear(xs)
        out = network_helper.addLinear(self.out_linear, xs)
        # out_lens = masks.sum(dim=-1).view(-1)

        if not output_embed:
            return out, xs_len
        else:
            return out, xs_len, xs
