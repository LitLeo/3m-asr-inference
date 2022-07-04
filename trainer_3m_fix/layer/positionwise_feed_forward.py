#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Positionwise feed forward layer definition."""

import torch
from fmoe.layers import FMoELinear
# from fmoe.functions import moe_prepare_forward
# from fmoe.functions import MOEScatter, MOEGather, MOELinear, MOEbiasLinear
# from loss.balance_loss import SparseL1Loss, BalanceImportanceLoss

import tensorrt as trt
import numpy as np

import trt_helper

# def mark_module_parallel_comm(m, dp_comm):
    # for p in m.parameters():
        # setattr(p, "dp_comm", dp_comm)


# def _fmoe_general_global_forward(inp, gate, expert_fn, num_expert, world_size,
                                 # capacity=-1, comm=None):
    # r"""
    # A private function that performs the following steps to complete the MoE
    # computation.
    # * Count the number of tokens from each worker to each expert.
    # * Send the features to their target position so that input features to each
    # expert are contiguous in memory.
    # * Perform the forward computation of the experts using `expert_fn`
    # * Gather the output features of experts back, and reorder them as sentences.
    # Intermediate results like expert counts are hidden from users by this
    # function.
    # """
    # (
        # pos,
        # local_expert_count,
        # global_expert_count,
        # fwd_expert_count,
        # fwd_batch_size,
    # ) = moe_prepare_forward(gate, num_expert, world_size, comm=comm)
    # x = MOEScatter.apply(
        # inp, pos,
        # local_expert_count, global_expert_count, fwd_batch_size, world_size
    # )
    # x = expert_fn(x, fwd_expert_count, capacity=capacity)
    # x = MOEGather.apply(
        # x, pos, local_expert_count, global_expert_count, inp.shape[0], world_size
    # )
    # return x


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """
    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: torch.nn.Module = torch.nn.ReLU()):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, network_helper, xs):
        xs = network_helper.addLinear(self.w_1, xs)

        # print(self.activation)
        # Swish ＝ x*sigmod(x)
        xs = network_helper.addSiLU(xs)

        xs = network_helper.addLinear(self.w_2, xs)

        return xs


class Expert(torch.nn.Module):
    def __init__(self,
                 num_experts: int,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: torch.nn.Module = torch.nn.ReLU(),
                 rank: int = 0):
        super(Expert, self).__init__()
        self.w_1 = FMoELinear(num_experts, idim, hidden_units, bias=True, rank=rank)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = FMoELinear(num_experts, hidden_units, idim, bias=True, rank=rank)

    def forward(self,
                xs: torch.Tensor,
                fwd_expert_count: torch.Tensor,
                capacity: float = -1.0) -> torch.Tensor:
        h = self.w_1(xs, fwd_expert_count, capacity=capacity)
        h = self.dropout(self.activation(h))
        h = self.w_2(h, fwd_expert_count, capacity=capacity)
        return h


class LocalFmoeCatEmbedFeedForward(torch.nn.Module):
    def __init__(self, idim, embed_dim, num_experts=4, rank=0, world_size=1, hidden_units=1024,
                 dropout_rate=0.0, activation=torch.nn.ReLU(), capacity_factor=-1.0,
                 router_regularization="l1_plus_importance", router_with_bias=False,
                 keep_expert_output=False, rand_init_router=False, comm=None):
        super(LocalFmoeCatEmbedFeedForward, self).__init__()
        self.rank = rank
        self.world_size = world_size
        self.comm = comm
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.router_regularization = router_regularization
        self.idim = idim
        self.hidden_units = hidden_units

        # expert network
        self.experts = Expert(num_experts, idim, hidden_units, dropout_rate,
                              activation=activation, rank=rank)
        # router
        router_input_dim = idim + embed_dim
        self.router_weights = torch.nn.Parameter(torch.zeros(router_input_dim, num_experts * world_size))
        if router_with_bias:
            self.router_bias = torch.nn.Parameter(torch.zeros(num_experts * world_size))
        else:
            self.router_bias = None

        # self.sparseLoss = SparseL1Loss(world_size)
        # self.balanceLoss = BalanceImportanceLoss(world_size)
        # if rand_init_router:
            # torch.nn.init.xavier_uniform_(self.router_weights, gain=0.5)
        self.keep_expert_output = keep_expert_output
        # mark_module_parallel_comm(self.experts, "mp")
        # setattr(self.router_weights, "dp_comm", "dp_mean")
        # if router_with_bias:
            # setattr(self.router_bias, "dp_comm", "dp_mean")

    # def gate(self, inputs):
        # router_logits = torch.einsum('ij,jk->ik', [inputs, self.router_weights])
        # if self.router_bias is not None:
            # router_logits = router_logits + self.router_bias
        # router_probs = torch.nn.functional.softmax(router_logits, dim=-1)
        # gate_value, gate_idx = router_probs.max(dim=-1)
        # all_samples = router_probs.size(0)
        # # router regularization
        # if self.router_regularization == "l1_plus_importance":
            # l1_loss, l1_loss_item, n_samples = self.sparseLoss(router_probs, group=self.comm)
            # all_samples = n_samples
            # importance_loss, importance_loss_item = self.balanceLoss(router_probs, group=self.comm)
            # aux_loss = ((l1_loss, l1_loss_item), (importance_loss, importance_loss_item))
        # else:
            # raise NotImplementedError("Not supported router regularization type: {}".format(
                                      # self.router_regularization))
        # return gate_idx, gate_value, aux_loss, all_samples

    def gate_trt(self, network_helper, inputs, mask):
        # print(self.router_weights.shape) 1024 32
        # print(inputs.shape) [-1, 1024]

        router_weights = self.router_weights.view(1, -1, self.num_experts)
        weight = network_helper.addConstant(router_weights)
        router_logits = network_helper.addMatMul(inputs, weight)
        bias = None
        if self.router_bias is not None:
            # self.router_weights = self.router_weights.view(1, -1, self.num_experts)
            bias = network_helper.addConstant(self.router_bias)
            router_logits = network_helper.addAdd(router_logits, bias)

        plugin_creator = network_helper.plugin_registry.get_plugin_creator("SoftmaxTopKPluginDynamic", "1", "")
        if not plugin_creator:
            raise RuntimeError("Could not find SoftmaxTopKPluginDynamic")

        # np_params = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 2, 0], dtype=np.int32)
        data_type = trt.PluginField("data_type", np.array([network_helper.config.plugin_data_type], dtype=np.int32), trt.PluginFieldType.INT32)
        # hidden_units = trt.PluginField("hidden_units", np.array([self.hidden_units], dtype=np.int32), trt.PluginFieldType.INT32)

        pfc = trt.PluginFieldCollection([data_type])
        plugin = plugin_creator.create_plugin("plugin", pfc)
        if not plugin:
            raise RuntimeError("Could not create_plugin SoftmaxTopKPluginDynamic")

        layer = network_helper.network.add_plugin_v2([router_logits, mask], plugin)

        network_helper.set_layer_name(layer, "SoftmaxTopKPluginDynamic")
        gate_value = layer.get_output(0)
        gate_idx = layer.get_output(1)

        # router_probs = network_helper.addSoftmax(router_logits, dim=-1)
        # gate_idx, gate_value = network_helper.addMax(router_probs, dim=-1)

        # network_helper.markOutput(router_probs)
        # network_helper.markOutput(gate_value)
        # gate_value = network_helper.addDumpTensor(gate_value)
        return gate_idx, gate_value

    def forward(self, network_helper, inputs, embed, mask):
        assert len(inputs.shape) == 3

        # batch_size = inputs.shape[0]
        # max_steps = inputs.shape[1]
        input_dim = inputs.shape[2]

        embed_dim = embed.shape[2]

        # inputs = inputs.view(-1, input_dim)
        # embed = embed.view(-1, embed_dim)
        # base_inputs = inputs
        # inputs = network_helper.addReshape(inputs, [-1, input_dim])
        # embed = network_helper.addReshape(embed, [-1, embed_dim])

        # router_inputs = torch.cat([embed, inputs], dim=-1)
        router_inputs = network_helper.addCat([embed, inputs], dim=-1)

        gate_idx, gate_value = self.gate_trt(network_helper, router_inputs, mask)
        # gate_value = network_helper.addDumpTensor(gate_value)
        # return base_inputs
        # import pdb
        # pdb.set_trace()

        fmoe_creator = network_helper.plugin_registry.get_plugin_creator("FMoEExpertPluginDynamic", "1", "")
        if not fmoe_creator:
            raise RuntimeError("Could not find FMoEExpertPluginDynamic")

        # np_params = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 2, 0], dtype=np.int32)
        data_type = trt.PluginField("data_type", np.array([network_helper.config.plugin_data_type], dtype=np.int32), trt.PluginFieldType.INT32)
        num_expert = trt.PluginField("num_expert", np.array([self.num_experts], dtype=np.int32), trt.PluginFieldType.INT32)
        idim = trt.PluginField("idim", np.array([self.idim], dtype=np.int32), trt.PluginFieldType.INT32)
        hidden_units = trt.PluginField("hidden_units", np.array([self.hidden_units], dtype=np.int32), trt.PluginFieldType.INT32)

        pfc = trt.PluginFieldCollection([data_type, num_expert, idim, hidden_units])
        plugin = fmoe_creator.create_plugin("plugin", pfc)
        if not plugin:
            raise RuntimeError("Could not create_plugin FMoEExpertPluginDynamic")

        w1_weight = network_helper.addConstant(self.experts.w_1.weight)
        w1_bias = network_helper.addConstant(self.experts.w_1.bias)
        w2_weight = network_helper.addConstant(self.experts.w_2.weight)
        w2_bias = network_helper.addConstant(self.experts.w_2.bias)
        layer = network_helper.network.add_plugin_v2([inputs, gate_idx, w1_weight, w1_bias, w2_weight, w2_bias], plugin)

        network_helper.set_layer_name(layer, "FMoEExpertPluginDynamic")
        expert_outputs = layer.get_output(0)

        if not self.keep_expert_output:
            expert_outputs = network_helper.addProd(expert_outputs, gate_value)
            # expert_outputs = network_helper.addProd(inputs, gate_value)

        output = expert_outputs
        # network_helper.markOutput(expert_outputs)
        # output = expert_outputs.view(batch_size, max_steps, -1)
        # output = network_helper.addReshapeLike(expert_outputs, base_inputs)
        return output
