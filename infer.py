import os
import time
import argparse
import ctypes
import numpy as np

from cuda import cudart
import tensorrt as trt

"""
TensorRT Initialization
"""
logger = trt.Logger(trt.Logger.VERBOSE)
# logger = trt.Logger(trt.Logger.INFO)

handle = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
if not handle:
    raise RuntimeError("Could not load plugin library. Is `libnvinfer_plugin.so` on your LD_LIBRARY_PATH?")

handle = ctypes.CDLL("libtrtplugin++.so", mode=ctypes.RTLD_GLOBAL)
if not handle:
    raise RuntimeError("Could not load plugin library. Is `libtrtplugin++_debug.so` on your LD_LIBRARY_PATH?")

trt.init_libnvinfer_plugins(logger, "")
plg_registry = trt.get_plugin_registry()

class InferHelper():
    """"""
    def __init__(self, plan_name, trt_logger):
        """"""
        self.logger = trt_logger
        self.runtime = trt.Runtime(trt_logger)
        with open(plan_name, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.context.active_optimization_profile = 0

    def infer(self, inputs: list):
        nInput = len(inputs)

        cudart.cudaDeviceSynchronize()
        _, stream = cudart.cudaStreamCreate()

        bufferD = []
        # alloc memory
        for i in range(nInput):
            _, inputD = cudart.cudaMallocAsync(inputs[i].nbytes, stream)
            bufferD.append(inputD)
            cudart.cudaMemcpyAsync(inputD, inputs[i].ctypes.data, inputs[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
            # cuda.memcpy_htod(bufferD[i], inputs[i].ravel())
            self.context.set_binding_shape(i, tuple(inputs[i].shape))
            # print(inputs[i].nbytes)

        # for i in range(0, self.engine.num_bindings):
            # print("get_binding_shape:" + str(self.context.get_binding_shape(i)))

        outputs = []
        for i in range(len(inputs), self.engine.num_bindings):
            outputs.append(np.zeros(self.context.get_binding_shape(i)).astype(np.float32))

        nOutput = len(outputs)
        for i in range(nOutput):
            _, outputD = cudart.cudaMallocAsync(outputs[i].nbytes, stream)
            bufferD.append(outputD)
            # bufferD.append(cuda.mem_alloc(outputs[i].nbytes))
            # print(outputs[i].nbytes)

        for i in range(len(inputs), self.engine.num_bindings):
            trt_output_shape = self.context.get_binding_shape(i)
            output_idx = i - len(inputs)
            if not (list(trt_output_shape) == list(outputs[output_idx].shape)):
                self.logger.log(trt.Logger.ERROR, "[Infer] output shape is error!")
                self.logger.log(trt.Logger.ERROR, "trt_output.shape = " + str(trt_output_shape))
                self.logger.log(trt.Logger.ERROR, "base_output.shape = " + str(outputs[output_idx].shape))
                assert(0)

        # warm up
        self.context.execute_async_v2(bufferD, stream)
        cudart.cudaStreamSynchronize(stream)

        T1 = time.perf_counter()

        self.context.execute_v2(bufferD)

        cudart.cudaStreamSynchronize(stream)
        T2 =time.perf_counter()
        print("time=" + str((T2-T1) * 1000) + "ms")

        for i in range(nInput, nInput + nOutput):
            # cuda.memcpy_dtoh(outputs[i - nInput].ravel(), bufferD[i])
            cudart.cudaMemcpyAsync(outputs[i - nInput].ctypes.data, bufferD[i], outputs[i - nInput].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

        for i in range(0, len(outputs)):
            print("outputs.shape:" + str(outputs[i].shape))
            print("outputs.sum:" + str(outputs[i].sum()))
            print(outputs[i])

            # print("trt_output.shape:" + str(trt_output.shape))
            # print("trt_output.sum:" + str(trt_output.sum()))
            # print(trt_output.view(-1)[0:10])
            # print("torch.allclose result:" + str(torch.allclose(base_output, trt_output, 1e-05, 1e-03)))
            # print("====================")
        return outputs
        # return torch.allclose(base_output, trt_output, 1e-05, 1e-03)


def main(args):
    # feat = np.load("feat.npy")
    # feat_len = np.load("feat_len.npy").reshape(1, -1)

    feat = np.load(args.input_file)
    feat_len = np.array([[0]], dtype=np.int32)
    feat_len[0][0] = feat.shape[1]

    plan_name = args.plan_name

    # print(feat_len)
    # print(feat_len.type)
    # assert 0
    infer_helper = InferHelper(plan_name, logger)
    output = infer_helper.infer([feat, feat_len])

    if args.compare_output_file:
        compare_output = np.load(args.compare_output_file)
        print(f"compare_output={args.compare_output_file}, dtype={str(compare_output.dtype)}, shape={str(compare_output.shape)}")
        print("output.sum:" + str(compare_output.sum()))
        print(compare_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch ASR --- inference to get AM score")

    parser.add_argument("-p", "--plan_name", required=True, help="The trt plan file path.")
    parser.add_argument("-i", "--input_file", required=True, help="The input feat.npy file path.")
    parser.add_argument("-o", "--compare_output_file", required=False, help="The compare output feat.npy file path.")

    args = parser.parse_args()
    main(args)
