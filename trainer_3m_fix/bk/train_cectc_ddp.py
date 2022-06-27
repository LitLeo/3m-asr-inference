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

from loss.loss_compute import MetricStat, CELoss, CTCLoss
from utils.lr_scheduler import build_optimizer
from utils.logger import set_logger
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP


cudnn.benchmark = False
cudnn.deterministic = True

MASTER_NODE = 0
world_size = int(os.environ["WORLD_SIZE"])


class LossCompute(nn.Module):
    def __init__(self, padding_idx, blank_idx, loss_type='ctc'):
        super(LossCompute, self).__init__()
        assert loss_type in ['ce', 'ctc']
        self.loss_type = loss_type
        if loss_type == 'ce':
            self.criterion = CELoss(padding_idx, blank_idx, mean_in_frames=False)
        else:
            self.criterion = CTCLoss(blank_idx, mean_in_batch=True)

    def compute_loss(self, output, target, lens, label_size, training=False):
        if self.loss_type == 'ce':
            loss, metric, count = self.criterion(output, target)
        else:
            loss, metric, count = self.criterion(output, lens, target, label_size)
        if training:
            loss.backward()
        return metric, count


class Trainer(object):
    def __init__(self, args, cfg):
        self.output_dir = args.output_dir
        self.local_rank = args.local_rank
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.cfg = cfg

        logger_name = "logger.{}".format(self.rank)
        log_file = args.log_file.replace("WORKER-ID", str(self.rank))
        self.log_f = set_logger(logger_name, log_file)

        self.make_loader(args)
        self.make_model()
        self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        # summary writer
        writer_dir = os.path.join(self.output_dir, "summary/rank%d" % self.rank)
        os.makedirs(writer_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=writer_dir)

        self.make_loss_obj()
        self.make_optimizer()
        self.make_checkpoint()

        self.stop_step = 0
        self.num_trained = 0

    def make_loader(self, args):
        self.tr_rspec = args.tr_rspecifier.replace("WORKER-ID", str(self.local_rank))
        self.tr_labels = args.tr_labels.replace("WORKER-ID", str(self.local_rank))
        self.cv_rspec = args.cv_rspecifier.replace("WORKER-ID", str(self.local_rank))
        self.cv_labels = args.cv_labels.replace("WORKER-ID", str(self.local_rank))
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

    def make_model(self):
        nnet_proto = self.cfg.get('nnet_proto')
        nnet_module = importlib.import_module("model." + nnet_proto)
        input_dim, output_dim = self.cfg['input_dim'], self.cfg['output_dim']
        self.model = nnet_module.Net(input_dim, output_dim, **self.cfg['model_conf'])
        # init model
        init_model = self.cfg.get('init_model', None)
        if init_model is None:
            self.log_f.info("Random initialize model")
        else:
            param_dict = torch.load(init_model, map_location="cpu")
            self.model.load_state_dict(param_dict)
            self.log_f.info("Initialize model from: {}".format(init_model))
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
        loss_type = self.cfg['loader_conf'].get('mode')
        assert loss_type in ['ce', 'ctc']
        if loss_type == 'ce':
            metric_tags = ['ce_loss', 'likely', 'acc']
            self.log_f.info("train model on CE")
        else:
            metric_tags = ['ctc_loss']
            self.log_f.info("train model on CTC")
        self.train_metric = MetricStat(metric_tags)
        self.valid_metric = MetricStat(metric_tags)
        self.train_loss_obj = LossCompute(padding_idx, blank_idx, loss_type=loss_type)
        self.valid_loss_obj = LossCompute(padding_idx, blank_idx, loss_type=loss_type)
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
            self.model.module.load_state_dict(param_dict)
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
            if self.rank == MASTER_NODE:
                torch.save(self.model.module.state_dict(), self.best_model)
                self.save_chkpt(self.start_epoch)

    def save_chkpt(self, epoch):
        chkpt_fn = "{}/chkpt".format(self.output_dir)
        optim_state = self.optimizer.state_dict()
        chkpt = {'epoch': epoch,
                 'best_model': self.best_model,
                 'best_valid_loss': self.best_valid_loss,
                 'recent_models': self.recent_models,
                 'global_step': self.global_step,
                 'optim': optim_state,
                 'resume_skip': self.epoch_num_trained}
        torch.save(chkpt, chkpt_fn)

    def save_model_state(self, epoch):
        cur_model = "{}/model.epoch-{}.step-{}".format(
                self.output_dir, epoch, self.global_step)
        if self.rank == MASTER_NODE:
            torch.save(self.model.module.state_dict(), cur_model)
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
        for (data, target, lens, label_lens) in self.train_loader(self.to_be_skip):
             # skip sentence only in 1st epoch and skip done when data ready
            # clear to_be_skip and synchronize the worker progresses
            if self.to_be_skip > 0:
                self.to_be_skip = 0
            # put data on corresponding GPU
            data = torch.from_numpy(data).cuda(self.local_rank)
            target = torch.from_numpy(target).long().cuda(self.local_rank)
            lens = torch.from_numpy(lens).cuda(self.local_rank)
            label_lens = torch.from_numpy(label_lens).cuda(self.local_rank)

            self.optimizer.zero_grad()
            batch_size = data.size(0)
            output, out_lens = self.model(data, lens)
            metrics, counts = self.train_loss_obj.compute_loss(
                    output, target, out_lens, label_lens, training=True)
            if schedule_type in ["warmup_linear", "warmup_cosine", "warmup_plateau", "warmup_noam"]:
                self.optimizer.addStep_adjustLR(1)
            elif schedule_type == "period_adjust":
                self.optimizer.addStep_adjustLR(batch_size)
            self.optimizer.step()
            self.train_metric.update_stat(metrics, counts)
            frames = torch.sum(lens).item()
            frames_log += frames
            self.num_trained += batch_size
            self.epoch_num_trained += batch_size
            self.global_step += 1
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
                cur_lr  = self.optimizer.get_learning_rate()
                self.log_f.info("Epoch: {},\tTrained sentences: {},\t"
                                "{}\tlr: {:.8f},\tfps: {:.1f} k".format(epoch,
                                self.num_trained, avg_str, cur_lr, frames_log/elapsed/1000))
                start_time = log_time
                frames_total += frames_log
                frames_log = 0
            # validation and save model
            if self.num_trained % valid_period < batch_size:
                # sync before valid
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
                    self.best_model = "{}/best_valid_model".format(self.output_dir)
                    if self.rank == MASTER_NODE:
                        os.system("cp {} {}".format(self.recent_models[-1], self.best_model))
                    self.log_f.info("new best_valid_loss: {}, storing best model: {}".format(
                                    self.best_valid_loss, self.recent_models[-1]))
                    self.stop_step = 0
                    if schedule_type == "cv_adjust":
                        self.optimizer.reset_step()
                else:
                    self.stop_step += 1
                    if schedule_type == "cv_adjust":
                        self.optimizer.addStep_adjustLR(1)
                if self.rank == MASTER_NODE:
                    self.save_chkpt(epoch)
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
        store_epoch_model = self.cfg.get('store_epoch_model', True)
        if store_epoch_model:
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

        for (data, target, lens, label_lens) in self.valid_loader(0):
            # put data on corresponding GPU device
            data = torch.from_numpy(data).cuda(self.local_rank)
            target = torch.from_numpy(target).long().cuda(self.local_rank)
            lens = torch.from_numpy(lens).cuda(self.local_rank)
            label_lens = torch.from_numpy(label_lens).cuda(self.local_rank)
            batch_size = data.size(0)
            with torch.no_grad():
                output, out_lens = self.model(data, lens)
                metrics, counts = self.valid_loss_obj.compute_loss(
                        output, target, out_lens, label_lens, training=False)
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
    parser.add_argument('--log_file', required=True, type=str,
            help='log file')
    parser.add_argument('--config', required=True, type=str,
            help='training yaml config file')
    parser.add_argument('--local_rank', type=int,
            help='local process ID for parallel training')

    args = parser.parse_args()
    main(args)
