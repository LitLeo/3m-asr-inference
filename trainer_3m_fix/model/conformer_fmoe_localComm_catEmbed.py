from typing import Tuple, List, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from model.conformer_embed import Net as ConformerEmbed
from layer.fmoe_transformer import FmoeConformerLayer
from layer.positionwise_feed_forward import LocalFmoeCatEmbedFeedForward
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


class Net(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        attention_heads: int = 4,
        attention_dim: int = 256,
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
        embed_conf = None,
        moe_conf = None
    ):
        """
        Conformer from Wenet Implementation
        """
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        activation = get_activation(activation_type)
        # embed module
        self.embed_conf = {
            'attention_heads': 4,
            'attention_dim': 512,
            'linear_units': 1024,
            'num_blocks': 6,
            'dropout_rate': 0.1,
            'positional_dropout_rate': 0.1,
            'attention_dropout_rate': 0.0,
            'input_layer': 'conv2d',
            'pos_enc_layer_type': 'rel_pos',
            'normalize_before': True,
            'concat_after': False,
            'static_chunk_size': 0,
            'use_dynamic_chunk': False,
            'use_dynamic_left_chunk': False,
            'positionwise_conv_kernel_size': 1,
            'macaron_style': True,
            'selfattention_layer_type': 'rel_selfattn',
            'activation_type': 'swish',
            'use_cnn_module': True,
            'cnn_module_kernel': 15,
            'causal': False,
            'cnn_module_norm': 'batch_norm',
            'conv_subsample_in_ch': 1
        }
        if embed_conf is not None and isinstance(embed_conf, dict):
            self.embed_conf.update(embed_conf)
        self.embed = ConformerEmbed(input_dim, output_dim, **self.embed_conf)
        embed_dim = self.embed_conf['attention_dim']
        # moe conf
        self.moe_conf = {
            'rank': 0,
            'world_size': 1,
            'comm': None,
            'num_experts': 4,
            'hidden_units': 1024,
            'dropout_rate': 0.0,
            'activation': activation,
            'capacity_factor': -1.0,
            'router_regularization': 'l1_plus_importance',
            'router_with_bias': False,
            'keep_expert_output': False,
            'rand_init_router': False
        }
        if moe_conf is not None:
            self.moe_conf.update(moe_conf)
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
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        # attention layer
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
        positionwise_layer = LocalFmoeCatEmbedFeedForward
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
            FmoeConformerLayer(
                attention_dim,
                selfattn_layer(*san_layer_args),
                positionwise_layer(attention_dim, embed_dim, **self.moe_conf),
                positionwise_layer(attention_dim, embed_dim, **self.moe_conf) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(num_blocks)
        ])
        # output layer
        self.out_linear = nn.Linear(attention_dim, self.output_dim)

    def init_embed_model(self, load_path):
        param_dict = torch.load(load_path, map_location='cpu')
        self.embed.load_state_dict(param_dict)

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        output_embed: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        # embed
        output_embed = True
        embed_out, _, embed, _ = self.embed(xs, xs_lens, output_embed=True)
        # detach embed to maintain an individual embeding
        embed = embed.detach()
        aux_loss_collection = []
        if not self.training:
            # set full chunk mode in validation
            decoding_chunk_size = -1
        masks = ~make_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, T)
        xs, masks = self.subsampling(xs, masks)
        xs, pos_emb = self.pos_enc(xs)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        for layer in self.blocks:
            xs, aux_loss_macaron, aux_loss, chunk_masks, _ = layer(
                    xs, embed, chunk_masks, pos_emb, mask_pad)
            if aux_loss_macaron is not None:
                aux_loss_collection.append(aux_loss_macaron)
            if aux_loss is not None:
                aux_loss_collection.append(aux_loss)
        if self.normalize_before:
            xs = self.after_norm(xs)
        out = self.out_linear(xs)
        out_lens = masks.sum(dim=-1).view(-1)
        if not output_embed:
            return out, out_lens, embed_out, aux_loss_collection
        else:
            return out, out_lens, embed_out, aux_loss_collection, xs, masks

    def state_dict_comm(self):
        local_state_dict = self.state_dict()
        rank = self.moe_conf['rank']
        world_size = self.moe_conf['world_size']
        num_experts = self.moe_conf['num_experts']
        comm = self.moe_conf['comm']
        if world_size <= 1:
            return local_state_dict
        else:
            new_state_dict = OrderedDict()
            all_experts_num = world_size * num_experts
            for k, v in local_state_dict.items():
                if "experts" not in k:
                    new_state_dict[k] = v
                else:
                    new_size = list(v.size())
                    new_size[0] = all_experts_num
                    experts_weight = v.data.new_zeros(*new_size)
                    experts_weight[rank * num_experts: (rank + 1) * num_experts] = v
                    dist.all_reduce(experts_weight, group=comm, async_op=False)
                    new_state_dict[k] = experts_weight
            return new_state_dict

    def load_state_dict_comm(self, whole_model_state):
        rank = self.moe_conf['rank']
        world_size = self.moe_conf['world_size']
        num_experts = self.moe_conf['num_experts']
        if world_size <= 1:
            return self.load_state_dict(whole_model_state)
        else:
            new_state_dict = OrderedDict()
            for k, v in whole_model_state.items():
                if "experts" not in k:
                    new_state_dict[k] = v
                else:
                    assert v.size(0) == num_experts * world_size
                    new_state_dict[k] = v[rank * num_experts: (rank + 1) * num_experts]
            return self.load_state_dict(new_state_dict)

