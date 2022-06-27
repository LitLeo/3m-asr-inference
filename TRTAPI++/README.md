[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

# TensorRT API ++
提高TensorRT使用效率的工具。
主要对两类开发者友好，
1. 不熟悉TRT，且使用现有TRT转换框架无法成功转换的开发者，比如算法人员和后台开发人员；
2. 非常熟悉TRT，想要对模型做深度合并的开发者。

对于以上两类问题，使用API方式构建TRT模型，是一个非常灵活有效的解决方案。
但TRT较高的学习成本和复杂的API调用方式，导致该解决方案费时费力。

为了解决以上问题，做了一些工作,包含两大部分，Torch-TensorRT-Plugin和TensorRT-Helper python api.

## TensorRT-Plugin ++
目的是实现TensorRT中不支持的Torch算子。用于解决因算子不支持导致现有TRT转换框架失败的问题。

### Build
```
cmake -DTRT_INSTALL_DIR="your/trt/path" -DTorch_INSTALL_DIR="your/torch/path" ..
```

### Prerequisites
To build this project, you will first need the following software packages.

**System Packages**
* TensorRT TensorRT == 7.2.2.3
* CUDA == cuda-10.2 + cuDNN-8.0
* libtorch == 1.8.0+cu102
* GCC >= 5.4.0, ld >= 2.26.1
* GNU make >= v4.1
* cmake >= v3.13


## TensorRT-Helper Python API
对TensorRT python api进行封装，与Torch 算子格式进行对齐，方便使用API的形式构建TRT模型。
以torch.nn.Conv1d算子举例，
模型代码如下：
```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_0 = nn.Conv1d(in_channels=20, out_channels=24, kernel_size=3, stride=1, padding=(4), dilation=1, groups=1, bias=False)

    def forward(self, x):
        x = self.conv_0(x)
        return x
```

直接调用TRT python api 构建模型的代码如下：
```python
weight = trt.Weights(self.conv_0.weight.detach().numpy())
bias = trt.Weights(self.conv_0.bias.detach().numpy()) if not self.conv_0.bias is None else None

trt_layer = network.add_convolution_nd(

x, num_output_maps=self.conv_0.out_channels,
kernel_shape=(1, self.conv_0.kernel_size[0]),
kernel=weight, bias=bias)

trt_layer.stride = (1, self.conv_0.stride[0])
trt_layer.padding = (0, self.conv_0.padding[0])
trt_layer.dilation = (1, self.conv_0.dilation[0])
trt_layer.num_groups = self.conv_0.groups
```

使用trt_helper api，代码如下：
```python
x = network_helper.addConv1d(self.conv_0, x, "conv_0")
```

