import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

SUCCESS=1
STOP=0

def _copy_vec_to_param(vec, parameters):
    r"""Copy vector to the parameters

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = param.data.copy_(vec[pointer:pointer + num_param]
                                      .view_as(param).data)
        # Increment the pointer
        pointer += num_param


class BmufTrainer(object):
    def __init__(self, model, dp_group, mp_group, dp_master_node,
                 world_master_node, local_rank, block_momentum, block_lr):
        self.model = model
        self.dp_group = dp_group
        self.mp_group = mp_group
        self.dp_master_node = dp_master_node
        self.master_node = world_master_node
        self.rank = dist.get_rank()
        self.local_rank = local_rank
        self.world_size = dist.get_world_size()
        self.dp_size = dist.get_world_size(group=dp_group)
        self.mp_size = dist.get_world_size(group=mp_group)
        assert self.world_size % self.dp_size == 0
        self.nproc_per_node = self.world_size // self.dp_size
        self.block_momentum = block_momentum
        self.block_lr = block_lr

        # dist init must be conducted before
        self.comm_types = ['dp_mean', 'dp_sum', 'mp']
        # get param list to be synced
        self.sync_params_world = []
        self.sync_params_dp = []
        for p in self.model.parameters():
            if hasattr(p, 'dp_comm') and p.dp_comm == 'mp':
                self.sync_params_dp.append(p)
            else:
                self.sync_params_world.append(p)
        param_vec_world = nn.utils.parameters_to_vector(self.sync_params_world)
        param_vec_dp = nn.utils.parameters_to_vector(self.sync_params_dp)
        self.param_world = param_vec_world.data.clone()
        self.param_dp = param_vec_dp.data.clone()
        # sync params before training
        dist.broadcast(self.param_world, src=self.master_node, async_op=False)
        dist.broadcast(self.param_dp, src=self.dp_master_node, group=self.dp_group, async_op=False)
        # block delta prev
        num_param_world = self.param_world.numel()
        if self.rank == self.master_node:
            self.delta_prev_world = torch.FloatTensor([0] * num_param_world).cuda(self.local_rank)
        else:
            self.delta_prev_world = None
            _copy_vec_to_param(self.param_world, self.sync_params_world)
        num_param_dp = self.param_dp.numel()
        if self.rank == self.dp_master_node:
            self.delta_prev_dp = torch.FloatTensor([0] * num_param_dp).cuda(self.local_rank)
        else:
            self.delta_prev_dp = None
            _copy_vec_to_param(self.param_dp, self.sync_params_dp)

    def update_and_sync(self):
        # data parallel group
        delta_dp = self.param_dp - nn.utils.parameters_to_vector(self.sync_params_dp).data
        dist.reduce(tensor=delta_dp, dst=self.dp_master_node, group=self.dp_group)
        if torch.isnan(delta_dp).sum().item():
            return STOP
        if self.rank == self.dp_master_node:
            delta_dp = delta_dp / float(self.dp_size)
            self.delta_prev_dp = self.block_momentum * self.delta_prev_dp + \
                                    self.block_lr * (1 - self.block_momentum) * delta_dp
            self.param_dp -= (1 + self.block_momentum) * self.delta_prev_dp
        dist.broadcast(tensor=self.param_dp, src=self.dp_master_node, group=self.dp_group)
        _copy_vec_to_param(self.param_dp, self.sync_params_dp)
        # world parallel
        delta = self.param_world - nn.utils.parameters_to_vector(self.sync_params_world).data
        dist.reduce(tensor=delta, dst=self.master_node)
        if torch.isnan(delta).sum().item():
            return STOP
        if self.rank == self.master_node:
            delta = delta / float(self.world_size)
            self.delta_prev_world = self.block_momentum * self.delta_prev_world + \
                                        self.block_lr * (1 - self.block_momentum) * delta
            self.param_world -= (1 + self.block_momentum) * self.delta_prev_world
        dist.broadcast(tensor=self.param_world, src=self.master_node)
        _copy_vec_to_param(self.param_world, self.sync_params_world)
        return SUCCESS

    def allreduce_grad(self):
        # sync grad within workers of same node
        groups = dict()
        for p in self.model.parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if hasattr(p, "dp_comm"):
                dp_comm = p.dp_comm
            else:
                dp_comm = "dp_mean"
            group_key = (dp_comm, p.dtype)
            if group_key not in groups:
                groups[group_key] = [p]
            else:
                groups[group_key].append(p)
        for (dp_comm, dtype), group in groups.items():
            if dp_comm not in self.comm_types:
                continue
            if dp_comm == "mp":
                continue
            grads = [p.grad.data for p in group]
            coalesced = _flatten_dense_tensors(grads)
            if dp_comm == "dp_mean":
                coalesced /= self.mp_size
            dist.all_reduce(coalesced, group=self.mp_group, async_op=False)
            synced = _unflatten_dense_tensors(coalesced, grads)
            for g, s in zip(grads, synced):
                g.copy_(s)

    def state_dict_comm(self):
        state_dict = {'block_momentum': self.block_momentum, 'block_lr': self.block_lr}
        if self.rank == self.master_node:
            state_dict['delta_prev_world'] = self.delta_prev_world
        # model parallel delta prev
        if self.rank == self.dp_master_node:
            pointer = 0
            reduce_deltas = []
            for param in self.sync_params_dp:
                numel = param.numel()
                param_delta = self.delta_prev_dp[pointer:pointer+numel].view_as(param).data
                pointer += numel
                new_size = list(param_delta.size())
                num_exp = new_size[0]
                new_size[0] = num_exp * self.nproc_per_node
                reduce_param_delta = param_delta.new_zeros(*new_size)
                reduce_param_delta[self.local_rank * num_exp: (self.local_rank + 1) * num_exp] = param_delta
                dist.all_reduce(reduce_param_delta, group=self.mp_group, async_op=False)
                reduce_param_delta = reduce_param_delta.cpu()
                reduce_deltas += [reduce_param_delta]
            delta_vec = nn.utils.parameters_to_vector(reduce_deltas)
            state_dict['delta_prev_dp'] = delta_vec
        return state_dict

    def load_state_dict_comm(self, state_dict):
        self.block_momentum = state_dict['block_momentum']
        self.block_lr = state_dict['block_lr']
        if self.rank == self.dp_master_node:
            delta_prev_dp = state_dict['delta_prev_dp']
            pointer = 0
            delta_list = []
            for param in self.sync_params_dp:
                numel = param.numel() * self.nproc_per_node
                new_size = list(param.size())
                num_exp = new_size[0]
                new_size[0] = num_exp * self.nproc_per_node
                reduce_param = delta_prev_dp[pointer: pointer+numel].view(new_size)
                pointer += numel
                delta_list += [reduce_param[self.local_rank*num_exp: (self.local_rank+1)*num_exp]]
            delta_dp_vec = nn.utils.parameters_to_vector(delta_list)
            self.delta_prev_dp.copy_(delta_dp_vec)
        if self.rank == self.master_node:
            self.delta_prev_world.copy_(state_dict['delta_prev_world'])

    def broadcast(self, tensor):
        dist.broadcast(tensor=tensor, src=self.master_node, async_op=False)

    def sum_reduce(self, tensor):
        dist.reduce(tensor=tensor, dst=self.master_node, async_op=False)
