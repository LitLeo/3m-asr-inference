from typing import Tuple, List, Optional, Dict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import BaseCTCEncoder
from layer.positionwise_feed_forward import PositionwiseFeedForward
from layer.transformer import ConformerEncoderLayer
from layer.attention import MultiHeadedAttention
from layer.attention import RelPositionMultiHeadedAttention
from layer.convolution import ConvolutionModule
from utils.mask import make_pad_mask
from utils.mask import add_optional_chunk_mask
from utils.common import get_activation
from loss.loss_compute import CTCLoss

import tensorrt as trt
import trt_helper

class Net(BaseCTCEncoder):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        blank_idx: int = 0,
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
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        conv_subsample_in_ch: int = 1,
        subsampling_feat_norm: bool = False,
        branch_blocks: Optional[int] = None
    ):
        super().__init__(
            input_dim, output_dim, blank_idx, attention_heads, attention_dim,
            linear_units, num_blocks, dropout_rate, positional_dropout_rate,
            attention_dropout_rate, input_layer, pos_enc_layer_type,
            normalize_before, concat_after, static_chunk_size,
            use_dynamic_chunk, use_dynamic_left_chunk,
            conv_subsample_in_ch, subsampling_feat_norm)
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
        if branch_blocks is None:
            self.branch_blocks = num_blocks * 2 // 3
        else:
            assert 0 < branch_blocks < num_blocks
            self.branch_blocks = branch_blocks
        self.blocks = nn.ModuleList([
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

    # def forward(
        # self,
        # xs: torch.Tensor,
        # xs_lens: torch.Tensor,
        # decoding_chunk_size: int = 0,
        # num_decoding_left_chunks: int = -1,
        # given_chunk_mask: Optional[torch.Tensor] = None
    # ) -> Dict[str, torch.Tensor]:
        # """
        # Args:
            # xs: padded input tensor (B, T, D)
            # xs_lens: input length (B)
            # decoding_chunk_size: decoding chunk size for dynamic chunk
                # 0: default for training, use random dynamic chunk.
                # <0: for decoding, use full chunk.
                # >0: for decoding, use fixed chunk size as set.
            # num_decoding_left_chunks: number of left chunks, this is for decoding,
            # the chunk size is decoding_chunk_size.
                # >=0: use num_decoding_left_chunks
                # <0: use all left chunks
            # given_chunk_mask: use consistent chunk_mask with another process
        # """
        # masks = ~make_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, T)
        # xs, masks = self.subsampling(xs, masks)
        # if self.subsampling_layer_norm is not None:
            # xs = self.subsampling_layer_norm(xs)
        # xs, pos_emb = self.pos_enc(xs)
        # mask_pad = masks  # (B, 1, T/subsample_rate)
        # if given_chunk_mask is not None:
            # chunk_masks = given_chunk_mask
        # else:
            # chunk_masks = add_optional_chunk_mask(
                    # xs, masks, self.use_dynamic_chunk,
                    # self.use_dynamic_left_chunk, decoding_chunk_size,
                    # self.static_chunk_size, num_decoding_left_chunks)
        # for i, layer in enumerate(self.blocks):
            # xs, chunk_masks, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
            # if i == self.branch_blocks - 1:
                # branch_xs = xs.clone()
        # if self.normalize_before:
            # xs = self.after_norm(xs)
        # out_nosm = self.out_linear(xs)
        # out_lens = masks.sum(dim=-1).view(-1)
        # res = {
            # "out_nosm": out_nosm,
            # "out_lens": out_lens,
            # "hidden": xs,
            # "branch_hidden": branch_xs
        # }
        # return res

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
            # network_helper.markOutput(xs)
            # break

        # xs = network_helper.addDumpTensor(xs, "transformer out")

        if self.normalize_before:
            # xs = self.after_norm(xs)
            xs = network_helper.addLayerNorm(self.after_norm, xs)

        # out = self.out_linear(xs)
        out = network_helper.addLinear(self.out_linear, xs)
        # out_lens = masks.sum(dim=-1).view(-1)

        if not output_embed:
            return out
        else:
            return out, xs_len, xs

