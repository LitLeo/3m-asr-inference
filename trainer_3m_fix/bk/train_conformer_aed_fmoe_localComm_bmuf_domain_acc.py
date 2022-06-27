import os
import sys
import time
import random
import yaml
import argparse
import traceback
import importlib
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from loss.loss_compute import MetricStat, CTCLoss, LabelSmoothingLoss, MoELayerScaleAuxLoss
from utils.lr_scheduler import build_optimizer
from utils.logger import set_logger
from torch.utils.tensorboard import SummaryWriter
# from torch.nn.parallel import DistributedDataParallel as DDP
from utils.fmoe_localComm_bmuf import BmufTrainer
from kaldi.util.table import SequentialMatrixReader, MatrixWriter
from kaldi.matrix import Matrix

cudnn.benchmark = False
cudnn.deterministic = True

MASTER_NODE = 0
world_size = int(os.environ["WORLD_SIZE"])


class LossCompute(nn.Module):
    def __init__(self, num_class, padding_idx, blank_idx, lsm_weight,
                 ctc_weight, ce_weight, reverse_weight, embed_scale, num_aux, aux_scale, auxloss_minimum,
                 normalize_length=False):
        super(LossCompute, self).__init__()
        self.ctc_criterion = CTCLoss(blank_idx, mean_in_batch=True)
        self.ce_criterion = nn.NLLLoss(ignore_index=padding_idx, reduction="sum")
        self.att_criterion = LabelSmoothingLoss(
                num_class, padding_idx, lsm_weight,
                normalize_length=normalize_length)
        self.ctc_weight = ctc_weight
        self.ce_weight = ce_weight
        self.reverse_weight = reverse_weight
        self.padding_idx = padding_idx
        assert 0.0 <= ctc_weight <= 1.0
        assert 0.0 <= ce_weight <= 1.0
        self.embed_scale = embed_scale
        self.aux_criterion = MoELayerScaleAuxLoss(num_aux, aux_scale, auxloss_minimum)

    def compute_loss(self, aux_loss, output, output_embed, lens, ctc_target,
                     ctc_lens, decoder_out, r_decoder_out, ys_out_pad, r_ys_out_pad, training=False):
        loss = 0.0
        metric, count = (), ()
        # CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, metric_ctc, count_ctc = self.ctc_criterion(
                    output, lens, ctc_target, ctc_lens)
            loss = loss + self.ctc_weight * loss_ctc
            metric += metric_ctc
            count += count_ctc
        # aed branch
        if self.ctc_weight != 1.0:
            loss_att, metric_att, count_att = self.att_criterion(
                    decoder_out, ys_out_pad)
            ###loss_att_6, metric_att_6, count_att_6 = self.att_criterion(
            ###        decoder_out_6, ys_out_pad)
            ###loss_att_12, metric_att_12, count_att_12 = self.att_criterion(
            ###        decoder_out_12, ys_out_pad)
            if self.reverse_weight > 0.0:
                r_loss_att, _, _ = self.att_criterion(
                        r_decoder_out, r_ys_out_pad)
                ###r_loss_att_6, _, _ = self.att_criterion(
                ###        r_decoder_out_6, r_ys_out_pad)
                ###r_loss_att_12, _, _ = self.att_criterion(
                ###        r_decoder_out_12, r_ys_out_pad)
                loss_att = loss_att * (1 - self.reverse_weight) + \
                            self.reverse_weight * r_loss_att
                ###loss_att_6 = loss_att_6 * (1 - self.reverse_weight) + \
                ###            self.reverse_weight * r_loss_att_6
                ###loss_att_12 = loss_att_12 * (1 - self.reverse_weight) + \
                ###            self.reverse_weight * r_loss_att_12
           ##### loss = loss + (1 - self.ctc_weight) * loss_att
            loss = loss + (1 - self.ctc_weight) * (loss_att)
            loss = loss * 0.0001
            metric += metric_att
            count += count_att
        # embed_branch
        if self.embed_scale > 0.0:
            embed_loss, embed_metric, embed_count = self.ctc_criterion(
                    output_embed, lens, ctc_target, ctc_lens)
            loss = loss + self.embed_scale * embed_loss
            metric += embed_metric
            count += embed_count
        
        # [B * T, D]
        ###flat_out_domain = output_domain.view(-1, output_domain.size(2))
        ###prob_domain = F.softmax(flat_out_domain, dim=-1)
        ###log_prob_domain = F.log_softmax(flat_out_domain, dim=-1)
        ######if ignore_sil:
        ######    target_domain[target_domain == self.sil_idx] = self.padding_idx
        ###target_domain = target_domain.view(-1)
        ###assert log_prob_domain.size(0) == target_domain.size(0)
        ###ce_loss = self.ce_criterion(log_prob_domain, target_domain)
        ###ce_loss = ce_loss / lens.size(0)
        #### likely
        ###mask = target_domain.ne(self.padding_idx)
        ###num_classes = prob_domain.size(1)
        ###frames = torch.sum(mask).item()
        ###true_prob = prob_domain[mask]
        ###true_target = target_domain[mask]
        #### hit
        ###prob_max = true_prob.argmax(dim=1)
        ###hit = torch.sum(true_target == prob_max).item()
        ###loss = loss + self.ce_weight * ce_loss

        #### [B * T, D]
        ###flat_out_acc = output_acc.view(-1, output_acc.size(2))
        ###prob_acc = F.softmax(flat_out_acc, dim=-1)
        ###log_prob_acc = F.log_softmax(flat_out_acc, dim=-1)
        ###target_acc = target_acc.view(-1)
        ###assert log_prob_acc.size(0) == target_acc.size(0)
        ###ce_loss_acc = self.ce_criterion(log_prob_acc, target_acc)
        ###ce_loss_acc = ce_loss_acc / lens.size(0)
        #### likely
        ###mask_acc = target_acc.ne(self.padding_idx)
        ###num_classes_acc = prob_acc.size(1)
        ###frames_acc = torch.sum(mask_acc).item()
        ###true_prob_acc = prob_acc[mask_acc]
        ###true_target_acc = target_acc[mask_acc]
        ###likely_acc = torch.sum(true_prob_acc * F.one_hot(true_target_acc, num_classes_acc).float())
        ###likely_acc = likely_acc.item()
        #### hit
        ###prob_max_acc = true_prob_acc.argmax(dim=1)
        ###hit_acc = torch.sum(true_target_acc == prob_max_acc).item()
        ###loss = loss  + self.ce_weight * ce_loss_acc

        ####metric += (hit,)
        ####metric += (hit_acc,)
        ####count += (frames,)
        ####count += (frames_acc,)


        # aux loss
        aux_loss, aux_metric, aux_count = self.aux_criterion(aux_loss)
        loss = loss + aux_loss
        metric += aux_metric
        count += aux_count
        if training:
            loss.backward(retain_graph=True)
        return metric, count

    def adjust_aux_scale(self, aux_metric):
        return self.aux_criterion.adjust_aux_scale(aux_metric)


class Trainer(object):
    def __init__(self, args, cfg):
        self.output_dir = args.output_dir
        self.local_rank = args.local_rank
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.cfg = cfg
        # arguments used by MoE model
        self.nproc_per_node = int(os.environ['nproc_per_node'])
        self.nnodes = int(os.environ['nnodes'])
        self.node_rank = int(os.environ['node_rank'])
        self.cfg['model_conf']['encoder_conf']['moe_conf']['rank'] = self.local_rank
        self.cfg['model_conf']['encoder_conf']['moe_conf']['world_size'] = self.nproc_per_node
        assert self.nnodes > 1 and self.world_size > 1
        self.build_groups()
        mp_group = self.mp_groups[self.node_rank]
        dp_group = self.dp_groups[self.local_rank]
        self.cfg['model_conf']['encoder_conf']['moe_conf']['comm'] = mp_group

        logger_name = "logger.{}".format(self.rank)
        log_file = args.log_file.replace("WORKER-ID", str(self.rank))
        self.log_f = set_logger(logger_name, log_file)

        self.make_loader(args)
        self.make_model()
        # bmuf trainer
        block_momentum = self.cfg.get('block_momentum', 0.0)
        block_lr = self.cfg.get('block_lr', 1.0)
        self.bmuf_trainer = BmufTrainer(self.model, dp_group, mp_group, self.local_rank,
                MASTER_NODE, self.local_rank, block_momentum, block_lr)
        # summary writer
        writer_dir = os.path.join(self.output_dir, "summary/rank%d" % self.rank)
        os.makedirs(writer_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=writer_dir)

        self.make_loss_obj()
        self.make_optimizer()
        self.make_checkpoint()

        self.stop_step = 0
        self.num_trained = 0

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

    def make_loader(self, args):
        self.tr_rspec = args.tr_rspecifier.replace("WORKER-ID", str(self.local_rank))
        self.tr_labels = args.tr_labels.replace("WORKER-ID", str(self.local_rank))
        self.cv_rspec = args.cv_rspecifier.replace("WORKER-ID", str(self.local_rank))
        self.cv_labels = args.cv_labels.replace("WORKER-ID", str(self.local_rank))
        self.tr_trans = args.tr_trans.replace("WORKER-ID", str(self.local_rank))
        # loader module
        loader_proto = self.cfg.get('loader_proto')
        loader_module = importlib.import_module("loader." + loader_proto)
        # collate func
        CollateFunc = loader_module.CollateFunc
        collate_func = CollateFunc(**self.cfg['collate_conf'])
        # loader
        self.train_loader = loader_module.DataLoader(
                self.tr_rspec,
                label_rspec=self.tr_labels,
                training=True,
                loader_conf=self.cfg['loader_conf'],
                collate_func=collate_func)
        self.valid_loader = loader_module.DataLoader(
                self.cv_rspec,
                label_rspec=self.cv_labels,
                training=False,
                loader_conf=self.cfg['loader_conf'],
                collate_func=collate_func)
        self.cfg['input_dim'] = self.train_loader.input_dim

    def init_embed_model(self, load_path):
        self.model.init_embed_model(load_path)
        self.log_f.info("Initialize embedding module from: {}".format(load_path))

    def init_experts_from_base(self, load_path):
        param_dict = torch.load(load_path, map_location='cpu')
        model_dict = self.model.state_dict()
        load_param_list = []
        for k, v in model_dict.items():
            if k in param_dict and param_dict[k].size() == v.size():
                model_dict[k] = param_dict[k]
                load_param_list.append(k)
            elif "experts" in k:
                ori_k = k.replace("experts.", "")
                if ori_k in param_dict and param_dict[ori_k] == v.size()[1:]:
                    model_dict[k] = param_dict[ori_k].unsqueeze(0).expand(v.size())
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
        num_experts = self.cfg['model_conf']['encoder_conf']['moe_conf']['num_experts']
        world_size = self.cfg['model_conf']['encoder_conf']['moe_conf']['world_size']
        all_experts = num_experts * world_size
        assert all_experts >= load_experts_num
        rank = self.cfg['model_conf']['encoder_conf']['moe_conf']['rank']
        start_idx = rank * num_experts
        end_idx = (rank + 1) * num_experts
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
            if k in model_dict and "router" in k:
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
        nnet_proto = self.cfg.get('nnet_proto')
        nnet_module = importlib.import_module("model." + nnet_proto)
        input_dim, output_dim = self.cfg['input_dim'], self.cfg['output_dim']
        padding_idx = self.cfg['loader_conf'].get('padding_idx', output_dim)
        self.model = nnet_module.Net(input_dim, output_dim,
            padding_idx=padding_idx, **self.cfg['model_conf'])
        # init model
        init_model = self.cfg.get('init_model', None)
        init_embed_model = self.cfg.get('init_embed_model', None)
        init_experts_from_base = self.cfg.get('init_experts_from_base', None)
        init_model_from_repeat = self.cfg.get('init_model_from_repeat', None)
        if init_embed_model is not None:
            self.init_embed_model(init_embed_model)
        if init_experts_from_base is not None:
            self.init_experts_from_base(init_experts_from_base)
        if init_model_from_repeat is not None:
            self.init_model_from_repeat(init_model_from_repeat)
        if init_model is None:
            self.log_f.info("Random initialize model")
        else:
            param_dict = torch.load(init_model, map_location='cpu')
            self.model.load_state_dict_comm(param_dict)
            self.log_f.info("Initialize model from: {}".format(init_model))


########################        else:
########################            param_dict = torch.load(init_model, map_location='cpu')
########################            model_dict = self.model.state_dict()
########################            miss_param_list = []
########################            for k, v in param_dict.items():
########################                if k in model_dict and model_dict[k].size() == v.size():
########################                    model_dict[k] = v
########################                else:
########################                    miss_param_list += [k]
            # parameter will be moved on model's device
            #####self.model.load_state_dict(model_dict)
            ###self.model.load_state_dict(param_dict)
            self.log_f.info("Initialize model from: {}".format(init_model))
            ####self.log_f.info("miss_match param list: {}".format(miss_param_list))

            #param_dict = torch.load(init_model, map_location='cpu')
            ################self.model.load_state_dict_comm(model_dict)
            #self.log_f.info("Initialize model from: {}".format(init_model))
        num_param = 0
        for param in self.model.parameters():
            num_param += param.numel()
        self.log_f.info("model proto: {},\tmodel_size: {} M".format(
            nnet_proto, num_param / 1000 / 1000))
        # place on gpu device
        self.model.cuda(self.local_rank)

    def make_loss_obj(self):
        padding_idx = self.cfg['loader_conf'].get('padding_idx')
        output_dim = self.cfg.get('output_dim')
        # use `output_dim - 1` as default
        blank_idx = self.cfg.get('blank_idx', output_dim - 1)
        lsm_weight = self.cfg.get('lsm_weight', 0.1)
        ctc_weight = self.cfg.get('ctc_weight', 0.3)
        ce_weight = self.cfg.get('ce_weight', 0.00001)
        reverse_weight = self.cfg.get('reverse_weight', 0.5)
        length_normalize_loss = self.cfg.get('length_normalize_loss', False)
        self.log_f.info("ctc_weight: {}, lsm_weight: {}".format(ctc_weight, lsm_weight))
        metric_tags = []
        if ctc_weight > 0.0:
            metric_tags += ['ctc_loss']
        if ctc_weight < 1.0:
            metric_tags += ['att_loss']
        embed_scale = self.cfg.get('embed_scale', 0.0)
        if embed_scale > 0.0:
            metric_tags += ['embed_ctc_loss']
        if ctc_weight < 1.0:
            metric_tags += ['domain_loss']
            metric_tags += ['acc_loss']
        # aux loss
        aux_tags = ['aux_loss']
        router_regularization = self.cfg['model_conf']['encoder_conf']['moe_conf'] \
                                    .get('router_regularization', 'l1_plus_importance')
        if router_regularization == "balance":
            aux_tags = ["balance_loss"]
        elif router_regularization == "l1_plus_importance":
            aux_tags = ["sparse_loss", "balance_loss"]
        num_aux = len(aux_tags)
        metric_tags += aux_tags
        aux_scale = self.cfg.get('aux_scale', None)
        if aux_scale is None:
            aux_scale = [0.1 for _ in range(num_aux)]
        auxloss_minimum = self.cfg.get('auxloss_minimum', None)
        self.train_metric = MetricStat(metric_tags)
        self.valid_metric = MetricStat(metric_tags)
        self.train_loss_obj = LossCompute(
                output_dim, padding_idx, blank_idx, lsm_weight, ctc_weight, ce_weight, reverse_weight,
                embed_scale, num_aux, aux_scale, auxloss_minimum,
                normalize_length=length_normalize_loss)
        self.valid_loss_obj = LossCompute(
                output_dim, padding_idx, blank_idx, lsm_weight, ctc_weight, ce_weight, reverse_weight,
                embed_scale, num_aux, aux_scale, auxloss_minimum,
                normalize_length=length_normalize_loss)
        self.train_loss_obj.cuda(self.local_rank)
        self.valid_loss_obj.cuda(self.local_rank)

    def make_optimizer(self):
        named_params = self.model.named_parameters()
        lr = self.cfg.get('lr', 1e-4)
        optim_type = self.cfg.get('optim')
        optim_conf = self.cfg.get('optim_conf', {})
        schedule_type = self.cfg.get('schedule_type')
        schedule_conf = self.cfg.get('schedule_conf', {})
        logger = self.log_f
        if 'name_nodecay' not in schedule_conf:
            schedule_conf['name_nodecay'] = [".bias", "Norm.weight"]
        self.optimizer = build_optimizer(named_params, schedule_type, schedule_conf,
                                         lr, optim_type, optim_conf, logger)
        self.log_f.info("[Optimizer] scheduler wrapper is {}".format(
            self.optimizer.__class__.__name__))

    def make_checkpoint(self):
        # TODO: save optimizer state for each worker requires collection
        chkpt_fn = "{}/chkpt".format(self.output_dir)
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
            self.best_model = "{}/model.epoch-0.step-0".format(self.output_dir)
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
        chkpt_fn = "{}/chkpt".format(self.output_dir)
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
                self.output_dir, epoch, self.global_step)
        model_state_dict = self.model.state_dict_comm()
        if self.rank == MASTER_NODE:
            torch.save(model_state_dict, cur_model)
        self.recent_models += [cur_model]
        num_recent_models = self.cfg.get('num_recent_models', -1)
        if num_recent_models > 0 and len(self.recent_models) > num_recent_models:
            pop_model = self.recent_models.pop(0)
            if self.rank == MASTER_NODE:
                os.remove(pop_model)

    def should_early_stop(self):
        early_stop_count = self.cfg.get('early_stop_count', 10)
        return self.stop_step >= early_stop_count

    def train_one_epoch(self, epoch):
        cur_lr = self.optimizer.get_learning_rate()
        self.log_f.info("Epoch {} start, lr {}".format(epoch, cur_lr))
        # by sentences
        log_period = self.cfg.get('log_period', 1000)
        valid_period = self.cfg.get('valid_period', 200000)
        schedule_type = self.cfg.get('schedule_type')
        # by global step
        sync_period = self.cfg.get('sync_period', 10)
        moe_node_sync = self.cfg.get('moe_node_sync', False)
        dynamic_aux_scale = self.cfg.get('dynamic_aux_scale', False)
        num_aux = len(self.train_loss_obj.aux_criterion.aux_scale)

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
            self.num_trained += self.to_be_skip
            self.epoch_num_trained += self.to_be_skip
        # run data
        for (data, target, aed_target, lens, label_lens, aed_lens, key_strs) in \
                self.train_loader(self.to_be_skip):
            # skip sentence only in 1st epoch and skip done when data ready
            # clear to_be_skip and synchronize the worker progresses
            if self.to_be_skip > 0:
                self.to_be_skip = 0
            # put data on corresponding GPU
            data = torch.from_numpy(data).cuda(self.local_rank)
            target = torch.from_numpy(target).long().cuda(self.local_rank)
            aed_target = torch.from_numpy(aed_target).long().cuda(self.local_rank)
            lens = torch.from_numpy(lens).cuda(self.local_rank)
            label_lens = torch.from_numpy(label_lens).cuda(self.local_rank)
            aed_lens = torch.from_numpy(aed_lens).cuda(self.local_rank)

            self.optimizer.zero_grad()
            batch_size = data.size(0)
            output, out_lens, decoder_out, ys_out_pad, r_decoder_out, r_ys_out_pad, output_embed, aux_loss = \
                    self.model(data, lens, aed_target, aed_lens)
            metrics, counts = self.train_loss_obj.compute_loss(
                    aux_loss, output, output_embed, out_lens, target,
                    label_lens, decoder_out, r_decoder_out, ys_out_pad, r_ys_out_pad, training=True)

            beam_grad = torch.zeros(output.shape[0], output.shape[1], output.shape[-1])
            beam_grad = beam_grad.cuda(self.local_rank)

            for utt_idx in range(output.shape[0]):
                tlg_fst = os.path.join(self.output_dir, "{}.trans.ark".format(self.rank))
                fst_trans = os.path.join(self.output_dir, "{}.trans.fst".format(self.rank))
                grep_cmd = "grep {} {} > {}".format(key_strs[utt_idx], self.tr_trans, fst_trans)
                os.system(grep_cmd)
                ##grep_time = time.time() - op_start_time
                ###op_start_time = time.time()
                tlg_cmd = "tlg ark:{} /apdcephfs/share_1297902/dennisyou/smbr_data/14whdata/smbr_src_word_wpe/L.fst /apdcephfs/share_1297902/dennisyou/smbr_data/14whdata/smbr_src_word_wpe/T.fst ark,t:{}".format(fst_trans, tlg_fst)
                os.system(tlg_cmd)
                ####tlg_time = time.time() - op_start_time
                ali_fst = os.path.join(self.output_dir, "{}.ali.ark".format(self.rank))
                lat_fst = os.path.join(self.output_dir, "{}.lat.ark".format(self.rank))
                prior_feat = os.path.join(self.output_dir, "{}.prior.ark".format(self.rank))
                grad_ark = os.path.join(self.output_dir, "{}.grad.ark".format(self.rank))
                ####op_start_time = time.time()
                posterior_prob = Matrix(output[utt_idx].contiguous().cpu().detach().numpy())
                with MatrixWriter("ark:{}".format(prior_feat)) as feat_writer:
                    feat_writer.write(key_strs[utt_idx], posterior_prob)
                #write_time = time.time() - op_start_time
                #op_start_time = time.time()
                ####cmd_online = "ctc-nnet-train-mpe-sequential-online --acoustic-scale_lat=1.0 --acoustic-scale_ali=1.0 --acoustic-scale=1.0 --lm-scale=1.0 --do-smbr=true --verbose=2 --one-silence-class=true --class-frame-counts=/apdcephfs/share_1297902/dennisyou/smbr_data/14whdata/smbr_src_word_wpe/train.counts --beam=10.0  --lattice-beam=8.0 --max-mem=50000000 --max-active=1000 --word-symbol-table=/apdcephfs/share_1297902/dennisyou/smbr_data/14whdata/smbr_src_word_wpe/words.txt ark:{} /apdcephfs/share_1297902/dennisyou/smbr_data/14whdata/smbr_src_word_wpe/TLG.fst ark:{} ark:{} ark:{}".format(prior_feat, tlg_fst, grad_ark, lat_fst)
                cmd_online = "ctc-nnet-train-mpe-sequential-online --acoustic-scale_lat=1.0 --acoustic-scale_ali=1.0 --acoustic-scale=1.0 --lm-scale=1.0 --do-smbr=true --verbose=2 --one-silence-class=true  --beam=10.0  --lattice-beam=8.0 --max-mem=50000000 --max-active=1000 --word-symbol-table=/apdcephfs/share_1297902/dennisyou/smbr_data/14whdata/smbr_src_word_wpe/words.txt ark:{} /apdcephfs/share_1297902/dennisyou/smbr_data/14whdata/smbr_src_word_wpe/ad_aishell_qq_spon.fst ark:{} ark:{} ark:{}".format(prior_feat, tlg_fst, grad_ark, lat_fst)
                os.system(cmd_online)
                #ctc_time = time.time() - op_start_time
                #op_start_time = time.time()
                with SequentialMatrixReader("ark:{}".format(grad_ark)) as grad_reader:
                    for key_grad, feats_grad in grad_reader:
                        beam_grad_utt = torch.from_numpy(feats_grad.numpy()).cuda(self.local_rank)
                        beam_grad[utt_idx] = beam_grad_utt
                        ###output.backward(beam_grad.unsqueeze(0))
                        ####output[utt_idx].backward(beam_grad)

            output.backward(beam_grad)



            if schedule_type in ["warmup_linear", "warmup_cosine", "warmup_plateau", "warmup_noam"]:
                self.optimizer.addStep_adjustLR(1)
            elif schedule_type == "period_adjust":
                self.optimizer.addStep_adjustLR(batch_size)
            if moe_node_sync:
                self.bmuf_trainer.allreduce_grad()
            self.optimizer.step()
            self.train_metric.update_stat(metrics, counts)
            frames = torch.sum(lens).item()
            frames_log += frames
            self.num_trained += batch_size
            self.epoch_num_trained += batch_size
            self.global_step += 1
            # sync among processes
            if self.global_step % sync_period == 0:
                self.update_and_sync()
                ###torch.cuda.empty_cache()
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
                cur_lr = self.optimizer.get_learning_rate()
                self.log_f.info("Epoch: {},\tTrained sentences: {},\t"
                                "{}\tlr: {:.8f},\tfps: {:.1f} k".format(epoch,
                                self.num_trained, avg_str, cur_lr, frames_log/elapsed/1000))
                start_time = log_time
                frames_total += frames_log
                frames_log = 0
            # validation and save model
            if self.num_trained % valid_period < batch_size:
                # sync before valid
                self.update_and_sync()
                ###valid_stat = self.valid()
                ###if self.rank == 0:
                ###    for tag, stat in zip(self.valid_metric.tags, valid_stat):
                ###        self.writer.add_scalar("valid/%s"%tag, stat, self.global_step)
                # save model state
                self.save_model_state(epoch)
                # check best loss
                ###valid_loss = valid_stat[0]
                ###if valid_loss < self.best_valid_loss:
                ###    self.best_valid_loss = valid_loss
                ###    self.best_model = "{}/best_valid_model".format(self.output_dir)
                ###    if self.rank == MASTER_NODE:
                ###        os.system("cp {} {}".format(self.recent_models[-1], self.best_model))
                ###    self.log_f.info("new best_valid_loss: {}, storing best model: {}".format(
                ###                    self.best_valid_loss, self.recent_models[-1]))
                ###    self.stop_step = 0
                ###    if schedule_type == "cv_adjust":
                ###        self.optimizer.reset_step()
                ###else:
                ###    self.stop_step += 1
                ###    if schedule_type == "cv_adjust":
                ###        self.optimizer.addStep_adjustLR(1)
                # all process will invoke the function to synchronize
                # only master node will save the file
                self.save_chkpt(epoch)
                ###torch.cuda.empty_cache()
                # back to train mode
                self.model.train()
                # adjust aux scale
                if dynamic_aux_scale:
                    valid_aux_metric = valid_stat[-num_aux:]
                    updated_scale = self.train_loss_obj.adjust_aux_scale(valid_aux_metric)
                    updated_scale = self.valid_loss_obj.adjust_aux_scale(valid_aux_metric)
                    self.log_f.info("adjust aux scale into: {}".format(updated_scale))
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
        store_epoch_model = self.cfg.get('store_epoch_model', True)
        if store_epoch_model:
            self.update_and_sync()
            self.save_model_state(epoch)
            epoch_model = "{}/model.epoch-{}".format(self.output_dir, epoch)
            if self.rank == MASTER_NODE:
                os.system("cp {} {}".format(self.recent_models[-1], epoch_model))
            # finished one epoch, reset epoch_num_trained and won't skip sentences next epoch
            self.epoch_num_trained = 0
            self.save_chkpt(epoch + 1)

    def valid(self):
        self.log_f.info("Start validation")
        log_period = 200
        num_sentences = 0
        self.model.eval()

        frames_total = 0
        frames_log = 0
        start_time = time.time()
        valid_start_time = start_time

        ###for (data, target, target_domain, target_acc, aed_target, lens, label_lens, aed_lens) in \
        for (data, target, aed_target, lens, label_lens, aed_lens) in \
                self.valid_loader(0):
            # put data on corresponding GPU device
            data = torch.from_numpy(data).cuda(self.local_rank)
            target = torch.from_numpy(target).long().cuda(self.local_rank)
            aed_target = torch.from_numpy(aed_target).long().cuda(self.local_rank)
            lens = torch.from_numpy(lens).cuda(self.local_rank)
            label_lens = torch.from_numpy(label_lens).cuda(self.local_rank)
            aed_lens = torch.from_numpy(aed_lens).cuda(self.local_rank)
            batch_size = data.size(0)
            with torch.no_grad():
                output, out_lens, decoder_out, ys_out, r_decoder_out, r_ys_out, output_embed, aux_loss = \
                        self.model(data, lens, aed_target, aed_lens)
                metrics, counts = self.valid_loss_obj.compute_loss(
                        aux_loss, output, output_embed, out_lens, target,
                        label_lens, decoder_out, r_decoder_out, ys_out, r_ys_out, training=False)
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
        dist.all_reduce(tensor=loss_tensor, async_op=False)
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

    def run(self):
        max_epochs = self.cfg.get('max_epochs')
        self.log_f.info("Start training")
        try:
            for epoch in range(self.start_epoch, max_epochs + 1):
                if self.should_early_stop():
                    self.log_f.info("Early stopping")
                    break
                self.train_one_epoch(epoch)
            self.log_f.info("Training Finished")
            if self.rank == MASTER_NODE:
                os.system("ln -s {} {}/final.nnet".format(
                    os.path.abspath(self.best_model), self.output_dir))
        except Exception as e:
            self.log_f.error("training exception: %s" % e)
            traceback.print_exc()


def init_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def main(args):
    # load config
    with open(args.config, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    # init distributed method
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    seed = configs.get('seed', 777)
    init_seed(seed + rank)
    # set default device
    torch.cuda.set_device(args.local_rank)
    trainer = Trainer(args, configs)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pytorch ASR training')

    parser.add_argument('--output_dir', required=True, type=str,
            help='path to save the final model')
    parser.add_argument('--tr_rspecifier', required=True, type=str,
            help='read specifier for training features')
    parser.add_argument('--tr_labels', required=True, type=str,
            help='read specifier for training labels')
    parser.add_argument('--cv_rspecifier', required=True, type=str,
            help='read rspecifier for cv features')
    parser.add_argument('--cv_labels', required=True, type=str,
            help='read rspecifier for cv labels')
    parser.add_argument('--tr_trans', required=True, type=str,
            help='read rspecifier for training trans')
    parser.add_argument('--log_file', required=True, type=str,
            help='log file')
    parser.add_argument('--config', required=True, type=str,
            help='training yaml config file')
    parser.add_argument('--local_rank', type=int,
            help='local process ID for parallel training')

    args = parser.parse_args()
    main(args)
