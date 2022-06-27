import os
import time
import random
import argparse
import traceback
import importlib
import re
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import utils.lr_scheduler as optim_wrapper
from utils.logger import set_logger
from torch.utils.tensorboard import SummaryWriter
from utils.fmoe_localComm_bmuf import BmufTrainer


cudnn.benchmark = False
cudnn.deterministic = True

FLAGS = None
MASTER_NODE = 0
world_size = int(os.environ["WORLD_SIZE"])


class LossCompute(nn.Module):
    """
    LossCompute class
    Args:
        - padding_idx: index used for padding label
        - blank_idx: index for blank unit
        - sil_idx: index for silence unit
        - loss_type: ctc or ce
    """
    def __init__(self, padding_idx, blank_idx, sil_idx, loss_type="ctc"):
        super(LossCompute, self).__init__()
        self.padding_idx = padding_idx
        self.blank_idx = blank_idx
        self.sil_idx = sil_idx
        self.loss_type = loss_type
        self.ce_criterion = nn.NLLLoss(ignore_index=padding_idx, reduction="sum")
        # add zero_infinity=True to ignore utterance with shorter length than label
        self.ctc_criterion = nn.CTCLoss(blank=blank_idx, reduction="sum",
                                        zero_infinity=True)

    def _compute_ce_loss(self, aux_loss, output, output_embed, target, ignore_sil=False):
        """
        Args:
            - output(FloatTensor): output from model, [B, T, D]
            - target(LongTensor): CE label, [B, T]
            - ignore_sil: whether ignore silence
        Return:
            - ce_loss: CE loss
            - metric: tuple of (CE loss, likelyhood, hit frames)
            - count: tuple of frames on each metric
        """
        # [B * T, D]
        flat_out = output.view(-1, output.size(2))
        flat_out_embed = output_embed.view(-1, output_embed.size(2))
        prob = F.softmax(flat_out, dim=-1)
        log_prob = F.log_softmax(flat_out, dim=-1)
        log_prob_embed = F.log_softmax(flat_out_embed, dim=-1)
        if ignore_sil:
            target[target == self.sil_idx] = self.padding_idx
        target = target.view(-1)
        assert log_prob.size(0) == target.size(0)
        assert log_prob_embed.size(0) == target.size(0)
        ce_loss = self.ce_criterion(log_prob, target)
        ce_loss_embed = self.ce_criterion(log_prob_embed, target)
        ce_loss_item = ce_loss.item()
        ce_loss_embed_item = ce_loss_embed.item()
        ce_loss = ce_loss + FLAGS.embed_scale * ce_loss_embed
        assert isinstance(aux_loss[0], tuple) and isinstance(aux_loss[0][0], tuple)
        num_aux_loss = len(aux_loss[0])
        aux_loss_sum = [0.0 for i in range(num_aux_loss)]
        aux_scale = FLAGS.aux_scale
        if aux_scale is None:
            aux_scale = [FLAGS.switch_scale for i in range(num_aux_loss)]
        for i in range(len(aux_loss)):
            for j in range(num_aux_loss):
                loss_tensor, loss_item = aux_loss[i][j][0], aux_loss[i][j][1]
                ce_loss = ce_loss + aux_scale[j] * loss_tensor
                aux_loss_sum[j] += loss_item
        # likely
        mask = target.ne(self.padding_idx)
        num_classes = prob.size(1)
        frames = torch.sum(mask).item()
        true_prob = prob[mask]
        true_target = target[mask]
        likely = torch.sum(true_prob * F.one_hot(true_target, num_classes).float())
        likely = likely.item()
        # hit
        prob_max = true_prob.argmax(dim=1)
        hit = torch.sum(true_target == prob_max).item()
        # print mean of frames on ce_loss, likely and acc
        metric = (ce_loss_item, ce_loss_embed_item, likely, hit, )
        count = (frames, frames, frames, frames, )
        for i in range(num_aux_loss):
            metric += (aux_loss_sum[i], )
            count += (1, )
        return ce_loss, metric, count

    #####def _compute_ctc_loss(self, aux_loss, output, output_embed, lens, label, label_size):
    def _compute_ctc_loss(self, aux_loss, output, output_embed, output_domain, output_acc, lens, label, target_domain, target_acc, label_size):
        """
        Args:
            - output(FloatTensor): output from model, [B, T, D]
            - lens(IntTensor): sequence length of output, [B]
            - label(LongTensor): ctc label, [B, T]
            - label_size(IntTensor): real length of label, [B]
        Returns:
            - ctc_loss: ctc loss
            - metric: tuple of (ctc loss,)
            - count: tuple of frames on each metric
        """
        # output shape is [B, T, D], should transform into [T, B, D]
        log_prob = F.log_softmax(output, dim=-1).transpose(0, 1).contiguous()
        log_prob_embed = F.log_softmax(output_embed, dim=-1).transpose(0, 1).contiguous()
        # output loss divided by target len and mean through batch
        ctc_loss = self.ctc_criterion(log_prob, label, lens, label_size)
        ctc_loss_embed = self.ctc_criterion(log_prob_embed, label, lens, label_size)
        loss_sum = ctc_loss.item()
        loss_embed_item = ctc_loss_embed.item()
        ctc_loss = ctc_loss / lens.size(0)
        ctc_loss_embed = ctc_loss_embed / lens.size(0)
        ctc_loss = ctc_loss + FLAGS.embed_scale * ctc_loss_embed
        # aux loss
        assert isinstance(aux_loss[0], tuple) and isinstance(aux_loss[0][0], tuple)
        num_aux_loss = len(aux_loss[0])
        aux_loss_sum = [0.0 for i in range(num_aux_loss)]
        aux_scale = FLAGS.aux_scale
        if aux_scale is None:
            aux_scale = [FLAGS.switch_scale for i in range(num_aux_loss)]
        for i in range(len(aux_loss)):
            for j in range(num_aux_loss):
                loss_tensor, loss_item = aux_loss[i][j][0], aux_loss[i][j][1]
                ctc_loss = ctc_loss + aux_scale[j] * loss_tensor
                aux_loss_sum[j] += loss_item
        # [B * T, D]
        flat_out_domain = output_domain.view(-1, output_domain.size(2))
        prob_domain = F.softmax(flat_out_domain, dim=-1)
        log_prob_domain = F.log_softmax(flat_out_domain, dim=-1)
        ###if ignore_sil:
        ###    target_domain[target_domain == self.sil_idx] = self.padding_idx
        target_domain = target_domain.view(-1)
        assert log_prob_domain.size(0) == target_domain.size(0)
        ce_loss = self.ce_criterion(log_prob_domain, target_domain)
        ce_loss = ce_loss / lens.size(0)
        # likely
        mask = target_domain.ne(self.padding_idx)
        num_classes = prob_domain.size(1)
        frames = torch.sum(mask).item()
        true_prob = prob_domain[mask]
        true_target = target_domain[mask]
        # hit
        prob_max = true_prob.argmax(dim=1)
        hit = torch.sum(true_target == prob_max).item()
        ctc_loss = ctc_loss + FLAGS.domain_scale * ce_loss

        # [B * T, D]
        flat_out_acc = output_acc.view(-1, output_acc.size(2))
        prob_acc = F.softmax(flat_out_acc, dim=-1)
        log_prob_acc = F.log_softmax(flat_out_acc, dim=-1)
        target_acc = target_acc.view(-1)
        assert log_prob_acc.size(0) == target_acc.size(0)
        ce_loss_acc = self.ce_criterion(log_prob_acc, target_acc)
        ce_loss_acc = ce_loss_acc / lens.size(0)
        # likely
        mask_acc = target_acc.ne(self.padding_idx)
        num_classes_acc = prob_acc.size(1)
        frames_acc = torch.sum(mask_acc).item()
        true_prob_acc = prob_acc[mask_acc]
        true_target_acc = target_acc[mask_acc]
        likely_acc = torch.sum(true_prob_acc * F.one_hot(true_target_acc, num_classes_acc).float())
        likely_acc = likely_acc.item()
        # hit
        prob_max_acc = true_prob_acc.argmax(dim=1)
        hit_acc = torch.sum(true_target_acc == prob_max_acc).item()
        ctc_loss = ctc_loss  + FLAGS.acc_scale * ce_loss_acc
        # print mean of ctc loss through utterances
        metric = (loss_sum, loss_embed_item, hit, hit_acc,)
        count = (lens.size(0), lens.size(0), frames, frames_acc,  )
        for i in range(num_aux_loss):
            metric += (aux_loss_sum[i], )
            count += (1, )
        return ctc_loss, metric, count

    ####def compute_loss(self, aux_loss, output, output_embed, target, lens, label_size,
    def compute_loss(self, aux_loss, output, output_domain, output_acc, output_embed, target, target_domain, target_acc, lens, label_size,
                     training=False, ce_ignore_sil=False):
        if self.loss_type == "ce":
            loss, metric, count = self._compute_ce_loss(aux_loss, output, output_embed, target,
                    ignore_sil=ce_ignore_sil)
        else:
            # CTC loss
            ####loss, metric, count = self._compute_ctc_loss(aux_loss, output, output_embed, lens, target, label_size)
            loss, metric, count = self._compute_ctc_loss(aux_loss, output, output_embed, output_domain, output_acc, lens, target, target_domain, target_acc, label_size)
        if training:
            loss.backward()
        return metric, count


class MetricStat(object):
    """
    Metric statistics class
    Args:
        - tags: name tag for each metric
    """
    def __init__(self, tags):
        super(MetricStat, self).__init__()
        self.tags = tags
        self.total_count = [0 for t in tags]
        self.total_sum = [0.0 for t in tags]
        self.log_count = [0 for t in tags]
        self.log_sum = [0.0 for t in tags]
        # for adjust
        self.adjust_count = [0 for t in tags]
        self.adjust_sum = [0.0 for t in tags]

    def update_stat(self, metrics, counts):
        for i, (m, c) in enumerate(zip(metrics, counts)):
            self.log_count[i] += c
            self.log_sum[i] += m
            # for adjust
            self.adjust_count[i] += c
            self.adjust_sum[i] += m

    def log_stat(self):
        """get recent average statistics"""
        avg = []
        for i, (m, c) in enumerate(zip(self.log_sum, self.log_count)):
            avg_stat = 0.0 if c == 0 else m / c
            avg += [avg_stat]
            self.total_sum[i] += m
            self.log_sum[i] = 0.0
            self.total_count[i] += c
            self.log_count[i] = 0
        return avg

    def report_adjust_stat(self):
        avg = []
        for i, (m, c) in enumerate(zip(self.adjust_sum, self.adjust_count)):
            avg_stat = 0.0 if c == 0 else m / c
            avg += [avg_stat]
            self.adjust_count[i] = 0
            self.adjust_sum[i] = 0.0
        return avg

    def summary_stat(self):
        """get total average statistics"""
        avg = []
        for i in range(len(self.tags)):
            self.total_sum[i] += self.log_sum[i]
            self.total_count[i] += self.log_count[i]
            avg_stat = 0.0
            if self.total_count[i] != 0:
                avg_stat = self.total_sum[i] / self.total_count[i]
            avg += [avg_stat]
        return avg

    def reset(self):
        for i in range(len(self.tags)):
            self.total_sum[i] = 0.0
            self.total_count[i] = 0
            self.log_sum[i] = 0.0
            self.log_count[i] = 0
            # for adjust
            self.adjust_sum[i] = 0.0
            self.adjust_count[i] = 0


class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.local_rank = self.cfg.local_rank
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # arguments used by MoE model
        self.cfg.expert_rank = self.local_rank
        self.nproc_per_node = int(os.environ['nproc_per_node'])
        self.cfg.world_size = int(os.environ['nproc_per_node'])
        self.nnodes = int(os.environ['nnodes'])
        self.node_rank = int(os.environ['node_rank'])
        assert self.nnodes > 1 and self.cfg.world_size > 1
        self.build_groups()
        mp_group = self.mp_groups[self.node_rank]
        dp_group = self.dp_groups[self.local_rank]
        self.cfg.mp_group = mp_group

        logger_name = "logger.{}".format(self.rank)
        self.log_f = set_logger(logger_name, cfg.log_file)

        self.make_loader()
        self.make_model()

        writer_dir = os.path.join(cfg.output_dir, "summary/rank%d"%self.rank)
        self.writer = SummaryWriter(log_dir=writer_dir)
        # self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        # self.model = DDP(self.model, self.world_size)
        mp_group = self.mp_groups[self.node_rank]
        dp_group = self.dp_groups[self.local_rank]
        self.bmuf_trainer = BmufTrainer(self.model, dp_group, mp_group, self.local_rank,
                MASTER_NODE, self.local_rank, cfg.block_momentum, cfg.block_lr)

        self.make_loss_obj()
        self.make_optimizer()
        self.make_checkpoint()

        self.stop_step = 0
        # trained sentences
        self.num_trained = 0
        torch.cuda.empty_cache()

    def make_loader(self):
        # just identify dataloader setup for training and validation
        loader_module = importlib.import_module("loader." + self.cfg.loader_proto)
        self.dataloader = loader_module.dataloader
        self.tr_rspec = self.cfg.tr_rspecifier.replace("WORKER-ID", str(self.local_rank))
        self.tr_labels = self.cfg.tr_labels.replace("WORKER-ID", str(self.local_rank))
        self.tr_labels_domain = self.cfg.tr_labels_domain.replace("WORKER-ID", str(self.local_rank))
        self.tr_labels_acc = self.cfg.tr_labels_acc.replace("WORKER-ID", str(self.local_rank))
        self.cv_rspec = self.cfg.cv_rspecifier.replace("WORKER-ID", str(self.local_rank))
        self.cv_labels = self.cfg.cv_labels.replace("WORKER-ID", str(self.local_rank))
        self.cv_labels_domain = self.cfg.cv_labels_domain.replace("WORKER-ID", str(self.local_rank))
        self.cv_labels_acc = self.cfg.cv_labels_acc.replace("WORKER-ID", str(self.local_rank))
        self.cfg.input_dim = loader_module.getInputDim(self.cfg)

    def build_groups(self):
        self.mp_groups = []
        for node_i in range(self.nnodes):
            mp_ranks = list(range(node_i * self.nproc_per_node, (node_i + 1) * self.nproc_per_node))
            mp_group = dist.new_group(ranks=mp_ranks)
            self.mp_groups += [mp_group]
        self.dp_groups = []
        for rank_i in range(self.nproc_per_node):
            dp_ranks = [rank_i + node_i * self.nproc_per_node for node_i in range(self.nnodes)]
            dp_group = dist.new_group(ranks=dp_ranks)
            self.dp_groups += [dp_group]

    def init_embed_model(self, load_path):
        param_dict = torch.load(load_path, map_location='cpu')
        self.model.embed.load_state_dict(param_dict)
        self.log_f.info("Initialize embedding module from: {}".format(load_path))

    def init_experts_from_base(self, load_path):
        param_dict = torch.load(load_path, map_location='cpu')
        model_dict = self.model.state_dict()
        load_param_list = []
        for k, v in model_dict.items():
            if "blocks_sw" in k:
                ori_k = k.replace("blocks_sw", "blocks")
                # non-expert layer
                if ori_k in param_dict and param_dict[ori_k].size() == v.size():
                    model_dict[k] = param_dict[ori_k]
                    load_param_list.append(k)
                # expert layer
                elif "experts" in ori_k:
                    ori_k = ori_k.replace("experts.", "")
                    if ori_k in param_dict and param_dict[ori_k].size() == v.size()[1:]:
                        model_dict[k] = param_dict[ori_k].unsqueeze(0).expand(v.size())
                        load_param_list.append(k)
            elif "out_linear_sw" in k:
                ori_k = k.replace("out_linear_sw", "out_linear")
                if ori_k in param_dict and param_dict[ori_k].size() == v.size():
                    model_dict[k] = param_dict[ori_k]
                    load_param_list.append(k)
        load_param_list.sort()
        self.model.load_state_dict(model_dict)
        self.log_f.info("Initialize experts model from baseline model: {}".format(load_path))
        self.log_f.info("following params have been initialized: {}".format(load_param_list))

    def init_model_from_repeat(self, load_path):
        param_dict = torch.load(load_path, map_location='cpu')
        load_experts_num = None
        for key in param_dict.keys():
            if "experts" in key and "weight" in key:
                load_experts_num = param_dict[key].size(0)
                break
        if load_experts_num is None:
            raise ValueError("Initailize model from MoE error: {} seems to have no experts".format(load_path))
        num_experts = self.cfg.num_experts
        world_size = self.cfg.world_size
        all_experts = num_experts * world_size
        assert all_experts >= load_experts_num
        start_idx = self.cfg.expert_rank * num_experts
        end_idx = (self.cfg.expert_rank + 1) * num_experts
        start_div = start_idx // load_experts_num
        start_rem = start_idx % load_experts_num
        end_div = end_idx // load_experts_num
        end_rem = end_idx % load_experts_num

        model_dict = self.model.state_dict()
        load_param_list = []
        # load parameters
        for k, v in param_dict.items():
            # non-expert layers
            if k in model_dict and "experts" not in k and model_dict[k].size() == v.size():
                model_dict[k] = v
                load_param_list.append(k)
            # expert layer
            if k in model_dict and "experts" in k:
                rest_dim = [1] * (v.dim() - 1)
                if end_div == start_div:
                    new_v = v[start_rem: end_rem]
                else:
                    new_v = v[start_rem:]
                    repeat = end_div - start_div - 1
                    new_v = torch.cat([new_v, v.repeat(repeat, *rest_dim)], dim=0)
                    new_v = torch.cat([new_v, v[0:end_rem]], dim=0)
                if model_dict[k].size() == new_v.size():
                    model_dict[k] = new_v
                load_param_list.append(k)
            # router
            if k in model_dict and "rooter" in k:
                dim = v.dim()
                assert dim <= 2
                if dim == 2:
                    model_dict[k][:, 0:load_experts_num] = v
                else:
                    model_dict[k][0:load_experts_num] = v
                load_param_list.append(k)
        load_param_list.sort()
        self.model.load_state_dict(model_dict)
        self.log_f.info("Initialize model from MoE model: {}".format(load_path))
        self.log_f.info("following params have been initialized: {}".format(load_param_list))

    def make_model(self):
        nnet_module = importlib.import_module("model." + self.cfg.nnet_proto)
        self.model = nnet_module.Net(self.cfg)
        # initialize embedding model
        if self.cfg.init_embed_model is not None:
            self.init_embed_model(self.cfg.init_embed_model)
        # initialize experts from baseline
        if self.cfg.init_experts_from_base is not None:
            self.init_experts_from_base(self.cfg.init_experts_from_base)
        # initialize experts from MoE model
        if self.cfg.init_model_from_repeat is not None:
            self.init_model_from_repeat(self.cfg.init_model_from_repeat)
        # initial model
        if self.cfg.init_model is None:
            self.log_f.info("Random initialize model")
        else:
            # load on cpu first
            param_dict = torch.load(self.cfg.init_model, map_location="cpu")
            self.model.load_state_dict_comm(param_dict)
            self.log_f.info("Initialize model from {}".format(FLAGS.init_model))
        num_param = 0
        for param in self.model.parameters():
            num_param += param.numel()
        # print model proto 
        self.log_f.info('*' * 60)
        self.log_f.info('*' * 60)
        self.log_f.info('*' * 60)
        self.log_f.info('model proto: {}'.format(self.cfg.nnet_proto))
        nnet_module.log_model_info(self.log_f, self.cfg)
        self.log_f.info('model size: {} M'.format(num_param/1000/1000))
        self.log_f.info('*' * 60)
        self.log_f.info('*' * 60)
        self.log_f.info('*' * 60)

        # model.cuda
        self.model.cuda(self.local_rank)
        ##self.model_embed.cuda(self.local_rank)

    def make_loss_obj(self):
        blank_idx = self.cfg.blank_idx
        padding_idx = self.cfg.padding_idx
        sil_idx = self.cfg.sil_idx
        loss_type = "ce" if self.cfg.pretrain_ce else "ctc"
        aux_tags = ["aux_loss"]
        if hasattr(self.cfg, "router_regularization"):
            if self.cfg.router_regularization == "balance":
                aux_tags = ["balance_loss"]
            elif self.cfg.router_regularization in ["l1_plus_importance"]:
                aux_tags = ["sparse_loss", "balance_loss"]
        # check reader mode
        if self.cfg.pretrain_ce:
            assert self.cfg.mode == "ce"
            metric_tags = ["ce_loss", "ce_loss_embed", "likely", "acc"] + aux_tags
            self.log_f.info("train model on CE")
        else:
            assert self.cfg.mode == "ctc"
            #####metric_tags = ["ctc_loss", "ctc_loss_embed"] + aux_tags
            metric_tags = ["ctc_loss", "ctc_loss_embed", "domain_acc", "accent_acc"] + aux_tags
            self.log_f.info("train model on CTC")
        self.train_metric = MetricStat(metric_tags)
        self.valid_metric = MetricStat(metric_tags)
        self.train_loss_obj = LossCompute(padding_idx, blank_idx, sil_idx, loss_type)
        self.valid_loss_obj = LossCompute(padding_idx, blank_idx, sil_idx, loss_type)
        self.train_loss_obj.cuda(self.local_rank)
        self.valid_loss_obj.cuda(self.local_rank)

    def make_optimizer(self):
        named_params = self.model.named_parameters()
        ###named_params = self.model.parameters() + self.model_embed.parameters()
        lr = self.cfg.lr
        optim_type = self.cfg.optim
        # add optimizer arguments
        kwargs = {}
        if optim_type == "sgd":
            kwargs["momentum"] = self.cfg.momentum
        schedule_type = self.cfg.schedule
        min_lr = self.cfg.min_lr
        logger = self.log_f
        grad_clip = self.cfg.grad_clip
        weight_decay = self.cfg.weight_decay
        name_nodecay = [".bias", "Norm.weight"]
        if schedule_type == "constant":
            self.optimizer = optim_wrapper.ConstantScheduleWrapper(
                    named_params, optim_type, lr, logger, min_lr=min_lr,
                    max_grad_norm=grad_clip, weight_decay=weight_decay,
                    name_nodecay=name_nodecay, **kwargs)
        elif schedule_type == "cv_adjust":
            self.optimizer = optim_wrapper.CVScheduleWrapper(
                    named_params, optim_type, lr, logger, min_lr=min_lr,
                    lr_decay=self.cfg.lr_decay, noImp_limit=self.cfg.lr_decay_count,
                    max_grad_norm=grad_clip, weight_decay=weight_decay,
                    name_nodecay=name_nodecay, **kwargs)
        elif schedule_type == "period_adjust":
            self.optimizer = optim_wrapper.PeriodScheduleWrapper(
                    named_params, optim_type, lr, logger, self.cfg.decay_period,
                    min_lr=min_lr, lr_decay=self.cfg.lr_decay,
                    max_grad_norm=grad_clip, weight_decay=weight_decay,
                    name_nodecay=name_nodecay, **kwargs)
        elif schedule_type == "warmup_linear":
            self.optimizer = optim_wrapper.WarmupLinearScheduleWrapper(
                    named_params, optim_type, lr, logger, min_lr=min_lr,
                    warmup=self.cfg.warmup, total_steps=self.cfg.total_steps,
                    max_grad_norm=grad_clip, weight_decay=weight_decay,
                    name_nodecay=name_nodecay, **kwargs)
        elif schedule_type == "warmup_cosine":
            self.optimizer = optim_wrapper.WarmupCosineScheduleWrapper(
                    named_params, optim_type, lr, logger, min_lr=min_lr,
                    warmup=self.cfg.warmup, total_steps=self.cfg.total_steps,
                    max_grad_norm=grad_clip, weight_decay=weight_decay,
                    name_nodecay=name_nodecay, **kwargs)
        elif schedule_type == "warmup_plateau":
            self.optimizer = optim_wrapper.WarmupPlateauScheduleWrapper(
                    named_params, optim_type, lr, logger, min_lr=min_lr,
                    t_step=self.cfg.t_step, d_step=self.cfg.d_step,
                    f_step=self.cfg.f_step, max_grad_norm=grad_clip,
                    weight_decay=weight_decay, name_nodecay=name_nodecay, **kwargs)
        else:
            raise NotImplementedError("Scheduler {} not Implemented!".format(schedule_type))
        self.log_f.info("[Optimizer] scheduler wrapper is {}".format(
            self.optimizer.__class__.__name__))

    def make_checkpoint(self):
        # TODO: save optimizer state for each worker requires collection
        chkpt_fn = "{}/chkpt".format(self.cfg.output_dir)
        if os.path.isfile(chkpt_fn):
            chkpt = torch.load(chkpt_fn, map_location="cpu")
            self.start_epoch = chkpt["epoch"]
            self.best_model = chkpt["best_model"]
            self.global_step = chkpt["global_step"]
            self.best_valid_loss = chkpt["best_valid_loss"]
            self.recent_models = chkpt["recent_models"]
            self.optimizer.load_state_dict(chkpt["optim"])
            self.to_be_skip = chkpt["resume_skip"]
            cur_lr = self.optimizer.get_learning_rate()
            self.optimizer.adjust_learning_rate(cur_lr)
            # load most recent model
            param_dict = torch.load(self.recent_models[-1], map_location="cpu")
            # DDP model
            self.model.load_state_dict_comm(param_dict)
            # load bmuf parameters
            self.bmuf_trainer.param_world.copy_(
                    nn.utils.parameters_to_vector(self.bmuf_trainer.sync_params_world).data)
            self.bmuf_trainer.param_dp.copy_(
                    nn.utils.parameters_to_vector(self.bmuf_trainer.sync_params_dp).data)
            self.bmuf_trainer.load_state_dict_comm(chkpt['bmuf'])
            self.log_f.info("loading checkpoint {} to continue training. "
                            "current lr is {}".format(chkpt_fn, cur_lr))
            self.log_f.info("loading most recent model from {}".format(
                            self.recent_models[-1]))
        else:
            self.log_f.info("no checkpoint, start training from scratch")
            self.start_epoch = 1
            self.best_model = "{}/model.epoch-0.step-0".format(FLAGS.output_dir)
            self.recent_models = [self.best_model]
            self.global_step = 0
            self.to_be_skip = 0
            self.epoch_num_trained = 0

            self.best_valid_loss = 1000000.0
            # only maintain single best_model, saved by MASTER_NDOE
            model_state_dict = self.model.state_dict_comm()
            if self.rank == MASTER_NODE:
                torch.save(model_state_dict, self.best_model)
            # all process will invoke this function to synchronize
            # only master_node will save the file
            self.save_chkpt(self.start_epoch)

    def save_chkpt(self, epoch):
        chkpt_fn = "{}/chkpt".format(self.cfg.output_dir)
        optim_state = self.optimizer.state_dict()
        bmuf_state = self.bmuf_trainer.state_dict_comm()
        chkpt = {'epoch': epoch,
                 'best_model': self.best_model,
                 'best_valid_loss': self.best_valid_loss,
                 'recent_models': self.recent_models,
                 'global_step': self.global_step,
                 'optim': optim_state,
                 'resume_skip': self.epoch_num_trained,
                 'bmuf': bmuf_state}
        if self.rank == MASTER_NODE:
            torch.save(chkpt, chkpt_fn)

    def save_model_state(self, epoch):
        cur_model = "{}/model.epoch-{}.step-{}".format(
                self.cfg.output_dir, epoch, self.global_step)
        model_state_dict = self.model.state_dict_comm()
        if self.rank == MASTER_NODE:
            torch.save(model_state_dict, cur_model)
        self.recent_models += [cur_model]
        if len(self.recent_models) > self.cfg.num_recent_models:
            pop_model = self.recent_models.pop(0)
            if self.rank == MASTER_NODE:
                os.remove(pop_model)

    def should_early_stop(self):
        return self.stop_step >= self.cfg.early_stop_count

    def adjust_aux_scale(self):
        minimum = 29
        avg_metric = self.train_metric.report_adjust_stat()
        metric_tensor = torch.FloatTensor(avg_metric).cuda(self.local_rank) / self.world_size
        dist.all_reduce(metric_tensor, async_op=False)
        avg_metric = metric_tensor.cpu().tolist()
        ####aux_metric = [max(m, minimum) for m in avg_metric[2:]]
        aux_metric = [max(m, minimum) for m in avg_metric[-2:]]
        if FLAGS.aux_scale is None:
            mean_metric = sum(aux_metric) / len(aux_metric)
            factor = min((mean_metric - minimum) / 10, 1.0)
            FLAGS.switch_scale = FLAGS.max_switch_scale * factor
            self.log_f.info("adjust switch_scale into {}".format(FLAGS.switch_scale))
        else:
            for i in range(len(aux_metric)):
                factor = min((aux_metric[i] - minimum) / 10, 1.0)
                FLAGS.aux_scale[i] = FLAGS.max_aux_scale[i] * factor
            self.log_f.info("adjust aux_scale into {}".format(FLAGS.aux_scale))

    def train_one_epoch(self, epoch):
        cur_lr = self.optimizer.get_learning_rate()
        self.log_f.info("Epoch {} start, lr {}".format(epoch, cur_lr))
        log_period = self.cfg.log_period
        valid_period = self.cfg.valid_period
        batch_size = self.cfg.batch_size

        self.epoch_num_trained = 0
        frames_total = 0
        frames_log = 0
        start_time = time.time()
        epoch_start_time = start_time
        # train mode
        self.model.train()
        # skip sentences needed by resume
        if self.to_be_skip > 0:
            self.log_f.info("should skip {} sentences...".format(self.to_be_skip))
            self.cfg.to_be_skip = self.to_be_skip
            self.num_trained += self.to_be_skip
            self.epoch_num_trained += self.to_be_skip
        # run data
        ###for (data, target, lens, label_lens) in \
        ###        self.dataloader(self.tr_labels, self.tr_rspec, True, self.cfg):
        for (data, target, target_domain, target_acc, lens, label_lens) in \
                self.dataloader(self.tr_labels, self.tr_labels_domain, self.tr_labels_acc, self.tr_rspec, True, self.cfg):
            # skip sentence only in 1st epoch and skip done when data ready
            # clear to_be_skip and synchronize the worker progresses
            if self.to_be_skip > 0:
                self.to_be_skip = 0
                self.cfg.to_be_skip = 0
                self.update_and_sync()
            # put data on corresponding GPU
            data = torch.from_numpy(data).cuda(self.cfg.local_rank)
            target = torch.from_numpy(target).long().cuda(self.cfg.local_rank)
            target_domain = torch.from_numpy(target_domain).long().cuda(self.cfg.local_rank)
            target_acc = torch.from_numpy(target_acc).long().cuda(self.cfg.local_rank)
            lens = torch.from_numpy(lens).cuda(self.cfg.local_rank)
            label_lens = torch.from_numpy(label_lens).cuda(self.cfg.local_rank)

            self.optimizer.zero_grad()
            # batch first
            if not self.cfg.batch_first:
                data = data.transpose(0, 1).contiguous()
                target = target.transpose(0, 1).contiguous()
                target_domain = target_domain.transpose(0, 1).contiguous()
                target_acc = target_acc.transpose(0, 1).contiguous()
            ####output, output_embed, out_lens, aux_loss = self.model(data, lens, True,
            output, output_domain, output_acc, output_embed, out_lens, aux_loss = self.model(data, lens, True,
                    stick_router=self.cfg.stick_router)
            # backward done when compute loss
            metrics, counts = self.train_loss_obj.compute_loss(aux_loss,
                    ####output, output_embed, target, out_lens, label_lens,
                    output, output_domain, output_acc, output_embed, target, target_domain, target_acc, out_lens, label_lens,
                    training=True, ce_ignore_sil=self.cfg.ce_ignore_sil)
            # schedule step
            if self.cfg.schedule in ["warmup_linear", "warmup_cosine", "warmup_plateau"]:
                self.optimizer.addStep_adjustLR(1)
            elif self.cfg.schedule == "period_adjust":
                self.optimizer.addStep_adjustLR(batch_size)
            self.optimizer.step()
            self.train_metric.update_stat(metrics, counts)
            frames = torch.sum(lens).item()
            frames_log += frames
            self.num_trained += batch_size
            self.epoch_num_trained += batch_size
            self.global_step += 1
            # sync among processes
            if self.global_step % self.cfg.sync_period == 0:
                self.update_and_sync()
            # log info 
            if self.num_trained % log_period < batch_size:
                log_time = time.time()
                elapsed = log_time - start_time
                avg_stat = self.train_metric.log_stat()
                avg_str = []
                for tag, stat in zip(self.train_metric.tags, avg_stat):
                    self.writer.add_scalar("train/%s"%tag, stat, self.global_step)
                    avg_str += ["{}: {:.6f},".format(tag, stat)]
                avg_str = '\t'.join(avg_str)
                self.log_f.info("Epoch: {},\tTrained sentences: {},\t"
                                "{}\tfps: {:.1f} k".format(epoch,
                                self.num_trained, avg_str, frames_log/elapsed/1000))
                start_time = log_time
                frames_total += frames_log
                frames_log = 0
            # adjust aux scale
            if self.cfg.mode == "ctc" and self.cfg.adjust_aux_period > 0 and \
                    self.num_trained % self.cfg.adjust_aux_period < batch_size:
                self.adjust_aux_scale()
            # validation and save model
            if self.num_trained % valid_period < batch_size:
                # sync before valid
                self.update_and_sync()
                valid_stat = self.valid()
                if self.rank == 0:
                    for tag, stat in zip(self.valid_metric.tags, valid_stat):
                        self.writer.add_scalar("valid/%s"%tag, stat, self.global_step)
                # save model state
                self.save_model_state(epoch)
                # check best loss
                valid_loss = valid_stat[0]
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.best_model = "{}/best_valid_model".format(self.cfg.output_dir)
                    if self.rank == MASTER_NODE:
                        os.system("cp {} {}".format(self.recent_models[-1], self.best_model))
                    self.log_f.info("new best_valid_loss: {}, storing best model: {}".format(
                                    self.best_valid_loss, self.recent_models[-1]))
                    self.stop_step = 0
                    if self.cfg.schedule == "cv_adjust":
                        self.optimizer.reset_step()
                else:
                    self.stop_step += 1
                    if self.cfg.schedule == "cv_adjust":
                        self.optimizer.addStep_adjustLR(1)
                # all process will invoke the function to synchronize
                # only master_node will save the file
                self.save_chkpt(epoch)
                torch.cuda.empty_cache()
                # back to train mode
                self.model.train()
        frames_total += frames_log
        train_stat = self.train_metric.summary_stat()
        self.train_metric.reset()
        avg_str = []
        for tag, stat in zip(self.train_metric.tags, train_stat):
            avg_str += ["{}: {:.6f},".format(tag, stat)]
        avg_str = '\t'.join(avg_str)
        elapsed = time.time() - epoch_start_time
        self.log_f.info("Epoch {} Done,\t{}\tAvg fps: {:.1f} k,"
                        "\tTime: {:.1f} hr,\t# frames: {:.1f} M".format(
                        epoch, avg_str, frames_total/elapsed/1000,
                        elapsed/3600, frames_total/1000/1000))
        if self.cfg.store_epoch_model:
            self.update_and_sync()
            self.save_model_state(epoch)
            epoch_model = "{}/model.epoch-{}".format(self.cfg.output_dir, epoch)
            if self.rank == MASTER_NODE:
                os.system("cp {} {}".format(self.recent_models[-1], epoch_model))
            # finished one epoch, reset epoch_num_trained and won't skip sentences next epoch
            self.epoch_num_trained = 0
            self.save_chkpt(epoch + 1)
        torch.cuda.empty_cache()

    def valid(self):
        self.log_f.info("Start validation")
        log_period = 200
        num_sentences = 0
        self.model.eval()

        frames_total = 0
        frames_log = 0
        batch_size = self.cfg.batch_size
        start_time = time.time()
        valid_start_time = start_time

        ###for (data, target, lens, label_lens) in \
        ###        self.dataloader(self.cv_labels, self.cv_rspec, False, self.cfg):
        for (data, target, target_domain, target_acc, lens, label_lens) in \
                self.dataloader(self.cv_labels, self.cv_labels_domain, self.cv_labels_acc,  self.cv_rspec, False, self.cfg):
            # put data on corresponding GPU device
            data = torch.from_numpy(data).cuda(self.cfg.local_rank)
            target = torch.from_numpy(target).long().cuda(self.cfg.local_rank)
            target_domain = torch.from_numpy(target_domain).long().cuda(self.cfg.local_rank)
            target_acc = torch.from_numpy(target_acc).long().cuda(self.cfg.local_rank)
            lens = torch.from_numpy(lens).cuda(self.cfg.local_rank)
            label_lens = torch.from_numpy(label_lens).cuda(self.cfg.local_rank)
            if not self.cfg.batch_first:
                data = data.transpose(0, 1).contiguous()
                target = target.transpose(0, 1).contiguous()
                target_domain = target_domain.transpose(0, 1).contiguous()
                target_acc = target_acc.transpose(0, 1).contiguous()
            with torch.no_grad():
                ####output, output_embed, out_lens, aux_loss = self.model(data, lens, False)
                output, output_domain, output_acc, output_embed, out_lens, aux_loss = self.model(data, lens, False)
                metrics, counts = self.valid_loss_obj.compute_loss(aux_loss,
                    ####output, output_embed, target, out_lens, label_lens,
                    output, output_domain, output_acc, output_embed, target, target_domain, target_acc, out_lens, label_lens,
                    training=False, ce_ignore_sil=self.cfg.ce_ignore_sil)
            self.valid_metric.update_stat(metrics, counts)
            frames = torch.sum(lens).item()
            frames_log += frames
            num_sentences += batch_size
            if num_sentences % log_period < batch_size:
                log_time = time.time()
                elapsed = log_time - start_time
                avg_stat = self.valid_metric.log_stat()
                avg_str = []
                for tag, stat in zip(self.valid_metric.tags, avg_stat):
                    avg_str += ["{}: {:.6f},".format(tag, stat)]
                avg_str = '\t'.join(avg_str)
                self.log_f.info("Valided Sentences: {},\t{}\t"
                                "fps: {:.1f} k".format(num_sentences,
                                avg_str, frames_log/elapsed/1000))
                frames_total += frames_log
                frames_log = 0
                start_time = log_time
        # finish validation
        frames_total += frames_log
        valid_stat = self.valid_metric.summary_stat()
        elapsed = time.time() - valid_start_time
        avg_str = []
        for tag, stat in zip(self.valid_metric.tags, valid_stat):
            avg_str += ["{}: {:.6f},".format(tag, stat)]
        avg_str = '\t'.join(avg_str)
        self.log_f.info("Validation Done,\t{}\tAvg fps: {:.1f} k,"
                        "\tTime: {} s\t# frames: {:.1f}M".format(
                        avg_str, frames_total/elapsed/1000, elapsed,
                        frames_total/1000/1000))
        # sync validation results
        tot_sum = self.valid_metric.total_sum
        tot_num = self.valid_metric.total_count
        loss_tensor = torch.FloatTensor([tot_sum, tot_num])
        loss_tensor = loss_tensor.cuda(self.local_rank)
        # sync valid metric
        dist.reduce(tensor=loss_tensor, dst=MASTER_NODE)
        dist.broadcast(tensor=loss_tensor, src=MASTER_NODE, async_op=False)
        self.valid_metric.reset()
        reduced_stat = loss_tensor[0] / loss_tensor[1]
        reduced_stat = reduced_stat.cpu().numpy()
        self.log_f.info("reduced valid loss: {}".format(reduced_stat[0]))
        return reduced_stat

    def update_and_sync(self):
        if not self.bmuf_trainer.update_and_sync():
            # model diverge
            self.log_f.warning("Model Diverges!")
            self.log_f.info("Reload {} and decay the "
                            "learning rate".format(self.best_model))
            # load parameter on cpu first
            param_dict = torch.load(self.best_model, map_location='cpu')
            self.model.load_state_dict_comm(param_dict)
            self.optimizer.half_learning_rate()
            self.stop_step += 1

    def run(self, num_epochs):
        try:
            self.log_f.info("Start training")
            for epoch in range(self.start_epoch, num_epochs + 1):
                if self.should_early_stop():
                    self.log_f.info("Early stopping")
                    break
                self.train_one_epoch(epoch)
            self.log_f.info("Training Finished")
            if self.rank == MASTER_NODE:
                os.system("ln -s {} {}/final.nnet".format(
                    os.path.abspath(self.best_model), self.cfg.output_dir))
        except Exception as e:
            self.log_f.error("training exception: %s" % e)
            traceback.print_exc()


def init_seed(seed):
    # manual seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    # init distributed method
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    init_seed(FLAGS.seed + rank)
    torch.cuda.set_device(FLAGS.local_rank)
    FLAGS.log_file = FLAGS.log_file.replace("WORKER-ID", str(FLAGS.local_rank))
    trainer = Trainer(FLAGS)
    trainer.run(FLAGS.max_epochs)
"""
train with multiple gpus on signle machine
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch ASR --- Deep-FSMN training')

    parser.add_argument("nnet_proto", type=str,
            help="pytorch NN proto definition filename")
    parser.add_argument("loader_proto", type=str,
            help="data loader definition filename")
    parser.add_argument("tr_rspecifier", type=str,
            help="read specifier of training features")
    parser.add_argument("tr_labels", type=str,
            help="label specifier of training")
    parser.add_argument("tr_labels_domain", type=str,
            help="domain label specifier of training")
    parser.add_argument("tr_labels_acc", type=str,
            help="acc label specifier of training")
    parser.add_argument("cv_rspecifier", type=str,
            help="read specifier of validation features")
    parser.add_argument("cv_labels", type=str,
            help="label specifier of validation")
    parser.add_argument("cv_labels_domain", type=str,
            help="domain label specifier of validation")
    parser.add_argument("cv_labels_acc", type=str,
            help="acc label specifier of validation")
    parser.add_argument("output_dir", type=str,
            help="path to save the final model")
    parser.add_argument("log_file", type=str,
            help="log file")
    parser.add_argument("--init_model", type=str, default=None,
            help="initial model")
    parser.add_argument("--init_embed_model", type=str, default=None,
            help="initial embedding model")
    parser.add_argument("--init_experts_from_base", type=str, default=None,
            help="initialize experts from base model")
    parser.add_argument("--init_model_from_repeat", type=str, default=None,
            help="initialize experts from other experts in repeated style")
    parser.add_argument("--stick_router", action='store_true',
            help="whether to keep router and embedding still")

    parser.add_argument("--blank_idx", type=int, default=1433,
            help="index of blank unit")
    parser.add_argument("--sil_idx", type=int, default=0,
            help="index of silence unit")
    parser.add_argument("--pretrain_ce", action="store_true",
            help="whether to pretrain CE")
    parser.add_argument("--ce_ignore_sil", action="store_true",
            help="whether to ignore silence unit when pretraining CE")
    # optimizer
    parser.add_argument("--optim", type=str, default="sgd",
            choices=["sgd", "adam", "adadelta"],
            help="optimizer to use")
    parser.add_argument("--schedule", type=str, default="warmup_linear",
            choices=["constant", "cv_adjust", "period_adjust",
            "warmup_linear", "warmup_cosine", "warmup_plateau"], help="lr scheduler method")
    parser.add_argument('--grad_clip', type=float, default=-1.0,
            help='gradient clipping threshold, valid when greater than zero')
    parser.add_argument("--weight_decay", type=float, default=0.0,
            help="l2 normalization for weight decay")
    parser.add_argument('--lr', type=float, default=1.0,
            help='initial learning rate')
    parser.add_argument('--switch_scale', type=float, default=0.05,
            help='switch_scale for SGD')
    parser.add_argument('--aux_scale', type=float, nargs='*', default=None,
            help='scales for auxiliary loss')
    parser.add_argument('--embed_scale', type=float, default=0.01,
            help='embed_scale for SGD')
    parser.add_argument('--domain_scale', type=float, default=0.1,
            help='domain_scale for SGD')
    parser.add_argument('--acc_scale', type=float, default=0.1,
            help='acc_scale for SGD')
    parser.add_argument('--momentum', type=float, default=0.9,
            help='momentum for SGD')
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--lr_decay_count', type=int, default=2)
    parser.add_argument("--min_lr", type=float, default=1e-8,
            help="minimum learning rate")
    parser.add_argument("--decay_period", type=int, default=1000000,
            help="decay learning rate after N sentences trained")
    # warmup
    parser.add_argument("--warmup", type=float, default=0.02,
            help="proportion of warming up steps")
    parser.add_argument("--total_steps", type=int, default=100000,
            help="total steps in warming up scheduler")
    parser.add_argument("--t_step", type=int, default=1000,
            help="at which step reach the peak lr")
    parser.add_argument("--d_step", type=int, default=40000,
            help="at which step start to decay lr")
    parser.add_argument("--f_step", type=int, default=160000,
            help="at which step decay to the final lr")

    parser.add_argument("--num_recent_models", type=int, default=5,
            help="save how many recent models")
    parser.add_argument('--max_epochs', type=int, default=10,
            help='max number of epochs for training')
    parser.add_argument("--log_period", type=int, default=1000,
            help="logging per n sentences")
    parser.add_argument("--valid_period", type=int, default=100000,
            help="validate the model per n sentences")
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--early_stop_count', type=int, default=8,
            help="if the model does not improve after "
            "early_stop_count times validation, stop training")

    parser.add_argument("--local_rank", type=int,
            help="local process ID for parallel training")
    parser.add_argument("--block_momentum", type=float, default=0.0,
            help="block momentum for BMUF")
    parser.add_argument("--block_lr", type=float, default=1.0,
            help="block learning rate for BMUF")
    parser.add_argument("--sync_period", type=int, default=100,
            help="synchronize among processes per sync_period step")

    parser.add_argument("--store_epoch_model", action="store_true",
            help="whether save model when each epoch finished")
    parser.add_argument("--adjust_aux_period", type=int, default=-1,
            help="adjust auxiliary loss scale after N sentences trained")

    FLAGS, unk = parser.parse_known_args()
    # add model config
    nnet_module = importlib.import_module("model." + FLAGS.nnet_proto)
    nnet_module.register(parser)
    # add reader config
    loader_module = importlib.import_module("loader." + FLAGS.loader_proto)
    loader_module.register(parser)
    FLAGS = parser.parse_args()
    # copy params
    FLAGS.max_aux_scale = copy.deepcopy(FLAGS.aux_scale)
    FLAGS.max_switch_scale = FLAGS.switch_scale

    main()
