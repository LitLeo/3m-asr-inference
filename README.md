# 3m-asr-inference

https://github.com/tencent-ailab/3m-asr

3m-asr模型是由Tencent AiLab提出的声学模型，由conformer+moe结构组成。
moe结构详见，https://github.com/laekov/fastmoe

moe结构是pytorch cuda extension实现的，使用现成的工具转换不方便，并且需要比较高的并行性。
选择使用api方式搭建+编写plugin，来转换加速这个模型。
