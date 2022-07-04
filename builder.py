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

import tensorrt as trt
import trt_helper

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

class ConformerConfig(trt_helper.HelperConfig):
    def __init__(self):
        super(ConformerConfig, self).__init__()

# logger = trt_helper.init_trt_plugin(trt.Logger.INFO, "libtorch_trt_plugin.so")
logger = trt_helper.init_trt_plugin(trt.Logger.VERBOSE, "libtrtplugin++.so")

def build_trt(model, args, input_dim, plan_name, prior = None):
    trt_config = ConformerConfig()
    trt_config.max_workspace_size = 8
    if args.fp16:
        trt_config.use_fp16 = True
        trt_config.plugin_data_type = trt.DataType.HALF

    if args.int8:
        assert 0
        trt_config.plugin_data_type = 1
        trt_config.use_int8 = True
        calibrator = AsrCalibrator("np_inputs/np_feat.list", "np_inputs/np_feat_len.list", "conformer.int8.cache", 10)

    calibrator = None
    # if trt_config.use_int8:

    builder_helper = trt_helper.BuilderHelper(trt_config, logger, calibrator)
    network_helper = builder_helper.get_network_helper()

    feat = network_helper.addInput(name="feat", dtype=trt.float32, shape=(-1, -1, input_dim))
    feat_len = network_helper.addInput(name="feat_len", dtype=trt.int32, shape=(1, -1))

    min_batch = 1
    opt_batch = 1
    max_batch = 1

    min_len = 1
    opt_len = 500
    max_len = 3000

    min_shape = (min_batch, min_len, input_dim)
    opt_shape = (opt_batch, opt_len, input_dim)
    max_shape = (max_batch, max_len, input_dim)

    builder_helper.add_profile("feat", min_shape, opt_shape, max_shape)
    builder_helper.add_profile("feat_len", (1, min_batch), (1, opt_batch), (1, max_batch))

    # builder_helper.builder_config.set_flag(trt.BuilderFlag.DEBUG)

    res = model.encoder(network_helper, feat, feat_len)

    if prior is not None:
        # score = score - np.log(prior)
        prior = -np.log(prior)
        torch_prior = torch.from_numpy(prior).float().view(1, 1, -1)
        trt_prior = network_helper.addConstant(torch_prior)
        res = network_helper.addAdd(res, trt_prior)

    # pdb.set_trace()
    network_helper.markOutput(res)

    engine = builder_helper.build_engine(plan_name)

    print("=======================bindings shape=====================")
    for i in range(engine.num_bindings):
        print("idx:" + str(i) + ", name: " + engine.get_binding_name(i) +", is_input: " + str(engine.binding_is_input(i)) + ", shape:" + str(engine.get_binding_shape(i)))
    print("=======================bindings shape=====================")

def main(args):

    # dist.init_process_group(backend="nccl", init_method="env://")
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    # torch.cuda.set_device(args.local_rank)

    with open(args.config, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    # configs['model_conf']['encoder_conf']['moe_conf']['rank'] = rank
    # configs['model_conf']['encoder_conf']['moe_conf']['world_size'] = world_size
    ###with open(args.config, 'r') as f:
    ###    configs = yaml.load(f, Loader=yaml.FullLoader)
    # make loader
    # test_rspec = args.test_rspec
    # loader_proto = configs.get('loader_proto')

    # loader_module = importlib.import_module("loader." + loader_proto)
    # loader_conf = configs.get('loader_conf')
    # loader_conf['batch_size'] = 1
    # loader_conf['cmvn_file'] = args.cmvn_file
    # CollateFunc = loader_module.CollateFunc
    # collate_func = CollateFunc(**configs['collate_conf'])

    # data_loader = loader_module.DataLoader(
            # test_rspec, label_rspec=None, training=False,
            # loader_conf=loader_conf, collate_func=collate_func)

    # configs['input_dim'] = data_loader.input_dim
    configs['input_dim'] = 40

    # make model
    nnet_proto = configs.get('nnet_proto')
    # print(nnet_proto)
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

    # read prior
    plan_name = args.output
    prior = None
    if args.prior_file:
        prior = read_prior(args.prior_file)

    build_trt(model, args, input_dim, plan_name, prior)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch ASR --- inference to get AM score")

    parser.add_argument("-m", "--load_path", required=False, help="The PyTorch checkpoint file path.")
    parser.add_argument("-o", "--output", required=True, default="bert_base_384.engine", help="The bert engine file, ex bert.engine")
    parser.add_argument("-c", "--config", required=True, help="config file")

    parser.add_argument("-prior", '--prior_file', required=False, help='prior file')
    parser.add_argument("-cmvn", '--cmvn_file', required=False, help='cmvn file')

    parser.add_argument("-f", "--fp16", action="store_true", help="Indicates that inference should be run in FP16 precision", required=False)
    parser.add_argument("-i", "--int8", action="store_true", help="Indicates that inference should be run in INT8 precision", required=False)
    parser.add_argument("-t", "--strict", action="store_true", help="Indicates that inference should be run in strict precision mode", required=False)
    parser.add_argument("-w", "--workspace-size", default=1000, help="Workspace size in MiB for building the BERT engine", type=int)

    parser.add_argument("-tcf", "--timing-cache-file", help="Path to tensorrt build timeing cache file, only available for tensorrt 8.0 and later", required=False)
    parser.add_argument("--verbose", action="store_true", help="Turn on verbose logger and set profiling verbosity to DETAILED", required=False)

    args = parser.parse_args()
    main(args)
