import os
import time
import struct
import argparse
import yaml
import importlib
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist

def read_prior(prior_file, minimum_prior=None):
    """
    read prior probability, smooth the zero value to minimum non-zero value
    """
    prior = np.loadtxt(prior_file)[1:]
    non_zero_min = min(prior[prior != 0])
    zero_idx = np.where(prior == 0)[0]
    prior[zero_idx] = non_zero_min * 1
    prior = prior / np.sum(prior)
    if minimum_prior is not None:
        prior = np.maximum(prior, minimum_prior)
    return prior


def write_score(score_path, score, prior):
    """write score for decoder to test"""
    ####score = score - np.log(prior)
    frames, dim = score.shape
    head_str = struct.pack('5i', 0, 0, 0, frames, dim)
    score = np.reshape(score, -1).tolist()
    out_str = struct.pack('%sf'%len(score), *score)
    out_str = head_str + out_str
    with open(score_path, 'wb') as f:
        f.write(out_str)


def main(args):
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(args.local_rank)
    with open(args.config, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    configs['model_conf']['encoder_conf']['moe_conf']['rank'] = rank
    configs['model_conf']['encoder_conf']['moe_conf']['world_size'] = world_size
    ###with open(args.config, 'r') as f:
    ###    configs = yaml.load(f, Loader=yaml.FullLoader)
    # make loader
    test_rspec = args.test_rspec
    loader_proto = configs.get('loader_proto')
    loader_module = importlib.import_module("loader." + loader_proto)
    loader_conf = configs.get('loader_conf')
    loader_conf['batch_size'] = 1
    loader_conf['cmvn_file'] = args.cmvn_file
    CollateFunc = loader_module.CollateFunc
    collate_func = CollateFunc(**configs['collate_conf'])
    data_loader = loader_module.DataLoader(
            test_rspec, label_rspec=None, training=False,
            loader_conf=loader_conf, collate_func=collate_func)
    configs['input_dim'] = data_loader.input_dim
    # make model
    nnet_proto = configs.get('nnet_proto')
    nnet_module = importlib.import_module("model." + nnet_proto)
    input_dim, output_dim = configs['input_dim'], configs['output_dim']
    model = nnet_module.Net(input_dim, output_dim, **configs['model_conf'])
    param_dict = torch.load(args.load_path, map_location='cpu')
    model.load_state_dict(param_dict)
    print("Loading model from {}".format(args.load_path))
    numel = 0
    for param in model.parameters():
        numel += param.numel()
    print("model parameter size: {}".format(numel))
    if torch.cuda.is_available() and args.cuda:
        model = model.cuda()
    # read prior
    prior = read_prior(args.prior_file)
    score_dir = args.output_dir
    if not os.path.exists(score_dir):
        os.makedirs(score_dir, exist_ok=True)
    # inference
    model.eval()
    start_time = time.time()
    skip_num = 0
    for data_item in data_loader(skip_num, output_keys=True):
        feat = data_item[0]
        keys_batch = data_item[-1]
        key_name = os.path.basename(keys_batch[0])
        feat = torch.from_numpy(feat)
        feat_len = torch.tensor([feat.size(1)]).int()
        if torch.cuda.is_available() and args.cuda:
            feat = feat.cuda()
            feat_len = feat_len.cuda()

        np.save("feat.npy", feat.cpu().numpy())
        np.save("feat_len.npy", feat_len.cpu().numpy())
        print(feat)
        print(feat_len)

        with torch.no_grad():
            res = model.forward_encoder(feat, feat_len)
        if isinstance(res, tuple):
            output = res[0]
        else:
            output = res
        if args.local_rank == 0:
            score = F.log_softmax(output, dim=-1).squeeze(0).cpu().data.numpy()
            score_path = os.path.join(score_dir, key_name)
            write_score(score_path, score, prior)
            print("[rank {}] Write score of {} into {}".format(args.local_rank, key_name, score_path))
            ###print("Write score of {} into {}".format(key_name, score_path))
        break
    duration = time.time() - start_time
    print("Inference and write score cost {} seconds".format(duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch ASR --- inference to get AM score")
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--load_path', required=True, help='load path')
    parser.add_argument('--test_rspec', required=True, help='test read specifier file')
    parser.add_argument('--prior_file', required=True, help='prior file')
    parser.add_argument('--cmvn_file', required=True, help='cmvn file')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--cuda', action='store_true', help='whether to use cuda')
    parser.add_argument('--local_rank', type=int, help='local process ID for parallel traininig')

    args = parser.parse_args()
    assert torch.cuda.is_available() and args.cuda
    main(args)
