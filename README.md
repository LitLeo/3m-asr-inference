# 3m-asr-inference

## 一、	总述
1. 模型名称：3M-ASR
2. 模型链接：https://github.com/tencent-ailab/3m-asr  ，Github上默认给出的模型是12层32e的，本校项目使用的是18层32e的。
该模型的运行环境搭建稍微麻烦一下，现成模型文件 链接：https://pan.baidu.com/s/18eM3DlYXLWU64cwbb2k7qw?pwd=jwos 
提取码：jwos

3. 优化效果（精度和加速比）：在T4显卡上，输入为206帧的真实语音数据，pytorch float time = 160.5ms, TensorRT float time = 20.44ms, 加速比7.x
4. Docker里代码的编译、运行步骤：


## 二、	原始模型
### 1、	模型简介
本项目的原始模型为WeNet模型的变种——3M-ASR，采用conformer+MoE结构，即将conformer的feed_forward结构替换成fast_moe结构，得益于MoE带来的模型容量提升，模型效果相比于原始conformer模型有了显著提升。
1）	用途以及效果：目前3M-ASR模型被应用于端到端的语音识别系统上，得益于FastMoE结构，不同专家处理不同token，对于多语种样本（例如粤语混合普通话），识别效果相比baseline具有显著提升
2）	业界运用情况：内部识别业务在用
3）	模型整体结构：模型整体结构采用conformer+moe，将conformer block中的feed_forward（一个FFN），替换为fast_moe结构（32个FFN），具体每个token由哪个FFN（专家）处理，由分发器（Gating）来控制。引入fast_moe结构结构后，token根据匹配度分发给不同的专家处理，模型学习到的参数更有代表性。fast_moe的结构详细见 https://github.com/laekov/fastmoe

### 2、	模型优化难点
由于引入了MoE结构，MoE结构是pytorch cuda extension实现的，不像普通的算子可以方便地通过PyTorch->ONNX->TensorRT模型转换的工作流来转换，且工具转换由于多了ONNX中间商，生成的ONNX格式的conformer模型结构非常复杂，不利于后续的深度优化，因此我们需要采用TRT python api+plugin的方式来构建trt模型（与TensorRT自带的demo/BERT类似），这样才能进行模型构建和有针对性的优化。
除此以外，在将feed_forward（一个FFN），替换为fast_moe结构（32个FFN）后，fast_moe层的参数量相比于普通feed_forward层增加了31倍，与此同时本项目模型具有18层fast_moe，导致模型的参数量非常大，需要对显存使用进行大量优化，同时32层FFN并行计算，需要对stream、cublas_handle等资源也要进行合理分配和优化。
综上，模型优化难点主要概括为以下两点：
1）	模型转换：需要采用TRT python api+plugin的方式来手动构建trt模型。而模型结构异常复杂+TRT API使用难度较高，导致转换过程工作量较大，效率低下。
2）	资源占用优化：在使用plugin实现fast_moe时，由于模型参数量巨大，显存不够用，导致build失败。需要对整个模型进行显存资源占用优化

## 三、	优化过程
### 1、	使用TensorRT python api搭建trt支持的模型部分
首先使用TensorRT python api搭建trt模型。
常见的TRT python api构建方式如TensorRT OOS BERT demo所示，构建流程为：读取ONNX/PyTorch/TF 模型权值的name和weight => 根据模型结构，根据name和weight向network中填充相应的算子 => 设置builder_config的参数并构建。
对于conformer+moe模型，该方法有两个缺点：（1）模型结构复杂算子众多，导致转换代码比较臃肿。如代码1，使用TRT API 填充 Conv层，所示。在conformer模型中，subsample和conv module结构中都有conv层，代码1就要写多遍。（2）业务迭代中，模型结构会根据配置发生更改，转换代码就需要及时维护更新。
针对以上问题，我提出了一种新的TRT api 构建模型的方式，两步：（1）将TRT api进行封装，向pytorch api看齐；（2）直接复用pytorch训练代码。废话不多说，直接上代码就理解了。选择conformer模型中PositionwiseFeedForward作为例子，代码2为训练代码，代码3 为trt api转换代码。

封装trt api可以提高编写转换代码的效率；复用pytorch训练代码可以提高代码的可维护性。

代码1： 使用TRT API 填充 Conv层
```
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

代码2 PositionwiseFeedForward 训练代码 
```python
class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: torch.nn.Module = torch.nn.ReLU()):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))
```

代码3 PositionwiseFeedForward trt api 转换代码
```
... 构造函数与训练代码相同
def forward(self, network_helper, xs):
    xs = network_helper.addLinear(self.w_1, xs)
    # Swish ＝ x*sigmod(x)
    xs = network_helper.addSiLU(xs)
    xs = network_helper.addLinear(self.w_2, xs)

    return xs

```

这个trt api 封装项目暂时叫TensorRT API++（欢迎大家推荐新名字），该工具目标是提高使用TensorRT API 构建模型的效率。目前正在内部使用，后续项目成熟时提供开源。

该工具将TensorRT Python API进行封装，提高其易用性，进而提高模型转换效率。具体地，（1）对TensorRT Python API进行封装，向PyTorch的算子API格式对齐，降低使用难度；（2）编写和收集Plugin算子集合，并提出了一种简单的、无需CUDA技能的Plugin编写方式，用于解决部分TensorRT API无法支持的PyTorch算子。

以torch.nn.Conv1d算子举例，
代码3， Conv1d算子模型代码如下：
```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_0 = nn.Conv1d(in_channels=20, out_channels=24, kernel_size=3, stride=1, padding=(4), dilation=1, groups=1, bias=False)

    def forward(self, x):
        x = self.conv_0(x)
        return x
```

直接调用TRT python api 构建模型如上面的代码1，
使用trt_helper api，只需要一行代码：

代码4 使用trt_helper api 构建conv 层
```python
x = network_helper.addConv1d(self.conv_0, x, "conv_0")
```

### 2、	自定义plugin实现trt不支持的部分
本部分主要涉及算子优化
#### 2.1 fast_moe plugin优化
fmoe_expert_plugin是本模型优化重点。如 fast_moe结构所示，https://github.com/laekov/fastmoe。多个expert是并行进行的，需要在plugin中使用cuda stream进行加速。
plugin的编写过程没什么难度，就是正常cuda stream编程。
出现最大的问题是显存不够用，T4显卡15G显存，build_engine过程中一直报错显存不够，缩小max_shape也不够。
##### 1) **经过单步调试发现，在填充网络过程中，显存占用一直在增大**
这个不正常，debug发现是trt7之后函数调用顺序+plugin编写方式导致。详细分析如下：
1. TensorRT-OSS 中提供的plugin（如embLayerNormPlugin， fcPlugin），资源（显存，句柄等）申请，自7.x之后，是放到了构造函数中。7.x之前我记得是放到initialize()函数中。比如代码5所示，embLayerNormPlugin的构造函数，copyToDevice函数即申请显存并copy权值。

代码5 embLayerNormPlugin的构造函数
```
EmbLayerNormVarSeqlenPluginBase::EmbLayerNormVarSeqlenPluginBase(std::string const& name, DataType const type,
    Weights const& beta, Weights const& gamma, Weights const& wordEmb, Weights const& posEmb, Weights const& tokEmb)
    : mLayerName(name)
    , mLd(beta.count)
    , mType(type)
{
    mWordVocabSize = wordEmb.count / mLd;
    mPosVocabSize = posEmb.count / mLd;
    mTokVocabSize = tokEmb.count / mLd;

    mBeta.convertAndCopy(beta, nvinfer1::DataType::kFLOAT);
    mGamma.convertAndCopy(gamma, nvinfer1::DataType::kFLOAT);

    mWordEmb.convertAndCopy(wordEmb, mType);
    mTokEmb.convertAndCopy(tokEmb, mType);
    mPosEmb.convertAndCopy(posEmb, mType);

    copyToDevice(mGamma, sizeof(float) * mGamma.count, mGammaDev);
    copyToDevice(mBeta, sizeof(float) * mBeta.count, mBetaDev);

    copyToDevice(mWordEmb, getWeightsSize(mWordEmb, mType), mWordEmbDev);
    copyToDevice(mPosEmb, getWeightsSize(mPosEmb, mType), mPosEmbDev);
    copyToDevice(mTokEmb, getWeightsSize(mTokEmb, mType), mTokEmbDev);
}
```

build阶段和infer阶段，trt plugin各个函数调用顺序如下
```
// build 阶段
1. Plugin::Plugin
2. Plugin::clone
3. Plugin::Plugin
4. Plugin::destroy
5. Plugin::clone
6. Plugin::Plugin
7. Plugin::clone
8. Plugin::Plugin
9. Plugin::clone
10. Plugin::Plugin
11. Plugin::destroy
12. Plugin::initialize
13. Plugin::destroy
14. Plugin::terminate
15. Plugin::destroy
16. Plugin::destroy

// infer 阶段
1. Plugin::deserialize_value
2. Plugin::initialize
3. Plugin::clone
4. Plugin::Plugin
5. Plugin::enqueue
6. Plugin::terminate
7. Plugin::destro
```

  - 发现的问题
1. plugin中申请的资源，期望在build or infer阶段中，只保留一份，否则就是浪费。
2. 资源申请是放到了构造函数中，每clone一个plugin类，就会申请一份资源。
3. 根据上述函数调用顺序可以发现，在build阶段，第11步 destroy之前，内存中存有四份资源。

  - 可能导致的严重后果
1. trt在build阶段，对显存的消耗本来就比较大。
2. 以我所做的一个3m-asr 32e模型为例，fmoe_expert层有两个矩阵乘，权值大小分别是32x512x1024和32x1024*512, 权值共128M，共有18层。那么会多额外占 18 * 128 * 3 = 6912M 显存。
3. 目前AI模型越来越大，合并的算子也越来越大，这个问题会愈发明显。

  - 尝试解决方案
对于权值比较好解决，将权值以输入的形式送入即可（比如groupNormalizationPlugin）。并强烈建议大家以后所有的权值全都以输入的形式送给plugin。

##### 2) **将权值以输入的形式送入后，发现填充网络过程中，显存还是会增长。**
继续debug发现，是在构造函数中初始化了cuda stream，cudaStreamCreate会申请显存，同样会存在上面浪费显存的问题。但cuda stream又不能以输入的形式送入。且trt plugin函数的调用顺序要求，资源必须在构造函数中申请(资源申请放到initialize里的话，infer阶段会在initialize后又clone一次……)

最终想到的折中解决方案是写三个构造函数，一个createPlugin时调用，一个clone时调用，一个deserialize时调用，资源申请放到一三中。将cuda stream等资源在clone时以参数的形式进行共享。

#### 2.2 mask优化
在pytorch源码中，mask是以二维bool矩阵[B, S]的形式存在。在(a) subsample逻辑；(b)attention模型 masked softmax逻辑；和(c)conv模块的mask_fill逻辑，使用mask时都需要花费一定算子进行格式转换。在本项目中，仿照TensorRT OOS BERT demo，将mask改为[B]维度，记录每句话的长度。为此分别实现了mask_conv2d_sample_plugin, att_masked_softmax_plugin和masked_fill_plugin，节省了大量转换算子。

#### 2.3 Layer_norm算子优化
Conformer中也涉及到了Layer_norm算子，在初赛后，我们学习了初赛复盘视频，从中学到了很多，其中就包括Layer_norm算子的优化。所以在本模型的Layer_norm算子部分，我们采用了大赛github官网，cookbook/05-Plugin/PluginReposity/LayerNormPlugin-V2.2-CUB-TRT8中的layer_norm实现方式，采用ReduceSum替代for循环，应用D(X) = E(X2) – E(X)2公式来计算方差，一趟归并既可计算公式右边两项，降低计算延迟。

#### 2.4 rel_positional_encoding_plugin
在conformer模型中的 RelPositionalEncoding 结构，如下代码，操作为：(a) 输入 乘 scale； (2)需要根据输入特征长度，截取pe。
```
x = x * self.xscale                                                                                            
pos_emb = self.pe[:, offset:offset + x.size(1)]                                                                
return self.dropout(x), self.dropout(pos_emb)
```
这里为了省事，合并成一个算子。

四、	精度与加速效果
还未进行FP16和INT8加速，但所有算子均支持FP16计算，并通过单元测试验证。


五、	Bug报告
https://github.com/NVIDIA/trt-samples-for-hackathon-cn/issues/32
