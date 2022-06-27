import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function


def compute_2d_balance_loss(router_probs, expert_mask, num_experts):
    router_probs = router_probs.view(-1, router_probs.size(-1))
    expert_mask = expert_mask.view(-1, expert_mask.size(-1))
    f_mean = torch.mean(expert_mask.float(), dim=0)
    p_mean = torch.mean(router_probs, dim=0)
    loss = torch.mean(f_mean * p_mean) * num_experts * num_experts
    return loss

def expect_activation_loss(router_probs, num_experts):
    router_probs = router_probs.view(-1, router_probs.size(-1))
    mean_gate = router_probs.mean(dim=0)
    diff = mean_gate - 1.0 / num_experts
    loss = diff.norm(1)
    return loss

def expert_importance_loss(router_probs, num_experts):
    router_probs = router_probs.view(-1, router_probs.size(-1))
    mean_gate = router_probs.mean(dim=0)
    important_loss = torch.sum(mean_gate * mean_gate) * num_experts
    return important_loss


def sparse_matrix_regularization(feat_matrix, eps=1e-20):
    assert feat_matrix.dim() >= 2
    feat_dim = feat_matrix.size(-1)
    flat_feat = feat_matrix.view(-1, feat_dim)
    n_samples = flat_feat.size(0)
    # l2 norm along sample axis
    norm_x = feat_matrix.norm(2, dim=-2, keepdim=True) + eps
    norm_feat = feat_matrix / norm_x
    # l2 norm along feat axis
    norm_y = norm_feat.norm(2, dim=-1, keepdim=True) + eps
    norm_feat = norm_feat / norm_y
    # l1 norm loss
    sparse_loss = norm_feat.norm(1)
    # avg on samples
    sparse_loss = sparse_loss / n_samples
    return sparse_loss


def sparse_l1_loss(feat_matrix, eps=1e-20):
    feat_matrix = feat_matrix.view(-1, feat_matrix.size(-1))
    n_samples = feat_matrix.size(0)
    norm_y = feat_matrix.norm(2, dim=-1, keepdim=True)
    norm_y = torch.clamp(norm_y, min=eps)
    norm_feat = feat_matrix / norm_y
    sparse_loss = norm_feat.norm(1) / n_samples
    return sparse_loss


class SparseL1Loss(nn.Module):
    def __init__(self, n_workers):
        super(SparseL1Loss, self).__init__()
        self.n_workers = n_workers

    def forward(self, prob_matrix, eps=1e-20, group=None):
        prob_matrix = prob_matrix.view(-1, prob_matrix.size(-1))
        n_samples =  prob_matrix.size(0)
        norm_y = prob_matrix.norm(2, dim=-1, keepdim=True)
        norm_y = torch.clamp(norm_y, min=eps)
        norm_prob = prob_matrix / norm_y
        l1_loss = norm_prob.norm(1)
        l1_loss_item = l1_loss.item()
        l1_loss = l1_loss / n_samples
        if self.n_workers > 1:
            loss_tensor = torch.FloatTensor([l1_loss_item, n_samples]).to(prob_matrix.device)
            dist.all_reduce(loss_tensor, group=group, async_op=False)
            n_samples = loss_tensor[1]
            l1_loss_item = loss_tensor[0]
        # l1_loss = l1_loss / n_samples
        l1_loss_item = l1_loss_item / n_samples
        return l1_loss, l1_loss_item, n_samples


class ReduceImportance(Function):
    @staticmethod
    def forward(ctx, prob_matrix, group=None):
        n_samples = prob_matrix.size(0)
        n_experts = prob_matrix.size(1)
        sum_prob = prob_matrix.sum(dim=0)
        # sync all frames
        pack_tensor = sum_prob.data.new_zeros(n_experts + 1)
        pack_tensor[0:-1] = sum_prob
        pack_tensor[-1] = n_samples
        dist.all_reduce(pack_tensor, group=group, async_op=False)
        sum_prob = pack_tensor[0:-1]
        n_samples = pack_tensor[-1]
        # compute loss
        mean_prob = sum_prob / n_samples
        importance_loss = torch.sum(mean_prob * mean_prob) * n_experts
        variables = (mean_prob,)
        ctx.balance_args = (n_samples, prob_matrix.size(0), n_experts)
        ctx.save_for_backward(*variables)
        return importance_loss

    @staticmethod
    def backward(ctx, grad_in):
        (mean_prob, ) = ctx.saved_tensors
        (total_num, this_num, n_experts) = ctx.balance_args
        grad_out = grad_in * mean_prob.unsqueeze(0).expand(this_num, -1) * 2
        # grad_out = grad_out * n_experts / total_num
        grad_out = grad_out * n_experts / this_num
        return grad_out, None


class BalanceImportanceLoss(nn.Module):
    def __init__(self, n_workers):
        super(BalanceImportanceLoss, self).__init__()
        self.n_workers = n_workers

    def forward(self, prob_matrix, group=None):
        prob_matrix = prob_matrix.view(-1, prob_matrix.size(-1))
        if self.n_workers > 1:
            importance_loss = ReduceImportance.apply(prob_matrix, group)
        else:
            n_experts = prob_matrix.size(1)
            mean_prob = prob_matrix.mean(dim=0)
            importance_loss = torch.sum(mean_prob * mean_prob) * n_experts
        return importance_loss, importance_loss.item()
