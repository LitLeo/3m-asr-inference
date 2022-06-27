import queue
import random
import torch
import numpy as np
from copy import copy
from threading import Thread
from kaldi.util.table import SequentialMatrixReader, SequentialIntVectorReader
from kaldi.transform.cmvn import Cmvn
from kaldi.feat.functions import DeltaFeaturesOptions, compute_deltas
from loader.utils import putThread, splice
from loader.augment import _spec_substitute, _spec_augmentation


class FeatureTransform(object):
    def __init__(self, feat_dim, order=2, window=2, cmvn_file=None):
        self.feat_dim = feat_dim
        self.delta_order = max(0, order)
        self.delta_window = max(0, window)
        self.delta_opt = DeltaFeaturesOptions(order=order, window=window)
        self.cmvn = None
        if cmvn_file is not None:
            self.cmvn = Cmvn()
            self.cmvn.read_stats(cmvn_file)
            assert self.cmvn.stats.size()[1] - 1 == self.dim

    @property
    def dim(self):
        return (1 + self.delta_order) * self.feat_dim

    def transform(self, feat):
        if self.delta_order > 0:
            feat = compute_deltas(self.delta_opt, feat)
        if self.cmvn is not None:
            self.cmvn.apply(feat, norm_vars=True)
        return feat


class CollateFunc(object):
    def __init__(self,
                 feature_dither=0.0,
                 spec_aug=False,
                 spec_aug_conf=None,
                 spec_sub=False,
                 spec_sub_conf=None):
        super(CollateFunc, self).__init__()
        self.spec_aug = spec_aug
        self.spec_aug_conf = spec_aug_conf
        self.spec_sub = spec_sub
        self.spec_sub_conf = spec_sub_conf
        self.feature_dither = feature_dither

    def __call__(self, feat, train_flag):
        if not train_flag:
            return feat
        if self.feature_dither != 0.0:
            a = random.uniform(0, self.feature_dither)
            feat = feat + (np.random.random_sample(feat.shape) - 0.5) * a
        if self.spec_sub:
            feat = _spec_substitute(feat, **self.spec_sub_conf)
        if self.spec_aug:
            feat = _spec_augmentation(feat, **self.spec_aug_conf)
        return feat

def NoneGenerator():
    while True:
        yield None

def fileLineGenerator(file_path):
    f = open(file_path, 'r')
    while True:
        line = f.readline()
        if line == "":
            break
        yield line
    f.close()


class DataLoader(object):
    def __init__(self,
                 feat_rspec,
                 label_rspec=None,
                 trans_file=None,
                 training=False,
                 loader_conf=None,
                 collate_func=None):
        super(DataLoader, self).__init__()
        self.feat_rspec = feat_rspec
        self.label_rspec = label_rspec
        self.trans_file = trans_file
        self.training = training
        self.collate_func = collate_func
        self.loader_conf = {
            "batch_size": 32,
            "feat_dim": 40,
            "queue_size": 30,
            "lctx": 0,
            "rctx": 0,
            "max_len": 1500,
            "add_deltas": 2,
            "delta_window": 2,
            "cmvn_file": None,
            "sampling": 1,
            "label_sampling": 1,
            "padding_idx": -1,
            "mode": 'ctc'
        }
        if loader_conf is not None and isinstance(loader_conf, dict):
            self.loader_conf.update(loader_conf)
        assert self.loader_conf['label_sampling'] in [1, 4, 6, 8]
        assert self.loader_conf['mode'] in ['ce', 'ctc']
        self.kaldi_feat_trans = FeatureTransform(
                self.loader_conf['feat_dim'],
                order=self.loader_conf['add_deltas'],
                window=self.loader_conf['delta_window'],
                cmvn_file=self.loader_conf['cmvn_file'])

    @property
    def input_dim(self):
        lctx = max(0, self.loader_conf['lctx'])
        rctx = max(0, self.loader_conf['rctx'])
        add_deltas = self.loader_conf['add_deltas']
        dim = self.loader_conf['feat_dim']
        if add_deltas > 0:
            dim *= (1 + add_deltas)
        if lctx > 0 or rctx > 0:
            dim *= (1 + lctx + rctx)
        return dim

    def __call__(self, skip_num, output_keys=False):
        queue_size = self.loader_conf['queue_size']
        q = queue.Queue(queue_size)
        thread = Thread(target=putThread, args=(q, self.produce, skip_num, output_keys))
        thread.setDaemon(True)
        thread.start()
        while True:
            item = q.get()
            q.task_done()
            if item is None:
                break
            yield item
        thread.join()

    def produce(self, skip_num, output_keys):
        feat_reader = SequentialMatrixReader(self.feat_rspec)
        if self.label_rspec is not None:
            label_reader = SequentialIntVectorReader(self.label_rspec)
        else:
            label_reader = NoneGenerator()
        if self.trans_file is not None:
            trans_reader = fileLineGenerator(self.trans_file)
        else:
            trans_reader = NoneGenerator()
        max_len = self.loader_conf['max_len']
        batch_size = self.loader_conf['batch_size']
        lctx, rctx = self.loader_conf['lctx'], self.loader_conf['rctx']
        sampling = self.loader_conf['sampling']
        label_sampling = self.loader_conf['label_sampling']
        padding_idx = self.loader_conf['padding_idx']
        mode = self.loader_conf['mode']

        data_buffer = np.zeros((batch_size, max_len, self.input_dim), dtype=np.float32)
        label_buffer = np.zeros((batch_size, max_len), dtype=np.int32)
        data_len_buffer = np.zeros(batch_size, dtype=np.int32)
        label_len_buffer = np.zeros(batch_size, dtype=np.int32)
        keys_batch = ['' for _ in range(batch_size)]
        trans_batch = ['' for _ in range(batch_size)]

        batch_idx = 0
        data_max_len = -1
        label_max_len = -1
        # skip sentences if needed
        done = False
        if skip_num > 0:
            while skip_num > 0:
                try:
                    feat_reader.next()
                    if self.label_rspec is not None:
                        label_reader.next()
                    else:
                        next(label_reader)
                    next(trans_reader)
                    skip_num -= 1
                except StopIteration:
                    done = True
                    break
        if done:
            yield None
            return
        # data generation
        for (uttid, feat), label_, trans_ in zip(feat_reader, label_reader, trans_reader):
            if label_ is not None:
                uttid2, label = label_
                assert uttid == uttid2
                label = np.array(label)
            else:
                label = None
            if trans_ is not None:
                trans_line = trans_
                uttid3 = trans_.strip().split()[0]
                assert uttid == uttid3
            else:
                trans_line = None
            feat = self.kaldi_feat_trans.transform(feat)
            feat = feat.numpy()
            # splice
            if lctx > 0 or rctx > 0:
                feat = splice(feat, lctx, rctx, pad=False)
            # sampling
            if sampling > 1:
                feat = feat[::sampling]
            # spec augment
            if self.collate_func is not None:
                feat = self.collate_func(feat, self.training)
            feat_len = feat.shape[0]
            data_len_buffer[batch_idx] = feat_len
            data_buffer[batch_idx, :feat_len, :] = feat
            data_max_len = max(data_max_len, feat_len)
            keys_batch[batch_idx] = uttid
            if trans_line is not None:
                trans_batch[batch_idx] = trans_line
            if label is not None:
                # ce
                if mode == 'ce':
                    if lctx > 0 or rctx > 0:
                        l_offset = (lctx + rctx) // 2
                        r_offset = lctx + rctx - l_offset
                        label = label[l_offset: -r_offset]
                    if sampling > 1:
                        label = label[::sampling]
                    assert label.shape[0] == feat_len
                    if label_sampling == 4:
                        label = label[1:-1:2]
                        label = label[1:-1:2]
                    elif label_sampling == 6:
                        label = label[1:-1:2]
                        label = label[2:-2:3]
                    elif label_sampling == 8:
                        label = label[1:-1:2]
                        label = label[1:-1:2]
                        label = label[1:-1:2]
                label_len = label.shape[0]
                label_len_buffer[batch_idx] = label_len
                label_buffer[batch_idx, :label_len] = label
                label_max_len = max(label_max_len, label_len)
            batch_idx += 1
            if batch_idx == batch_size:
                for b in range(batch_size):
                    feat_len = data_len_buffer[b]
                    data_buffer[b, feat_len: data_max_len, :] = 0
                    if label is not None:
                        label_len = label_len_buffer[b]
                        label_buffer[b, label_len: label_max_len] = padding_idx
                data = np.copy(data_buffer[:, :data_max_len, :])
                data_lens = np.copy(data_len_buffer)
                if label is not None:
                    target = np.copy(label_buffer[:, :label_max_len])
                    target_lens = np.copy(label_len_buffer)
                else:
                    target = None
                    target_lens = None
                if self.trans_file is not None:
                    yield data, target, data_lens, target_lens, copy(trans_batch)
                elif not output_keys:
                    yield data, target, data_lens, target_lens
                else:
                    yield data, target, data_lens, target_lens, copy(keys_batch)

                batch_idx = 0
                data_max_len = -1
                label_max_len = -1
        yield None

