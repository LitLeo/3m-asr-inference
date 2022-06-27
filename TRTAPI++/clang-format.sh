#!/bin/bash

plugins="att_masked_softmax_plugin cat_split_cache_plugin glu_plugin layer_norm_plugin masked_fill_plugin \
silu_plugin att_stream_softmax_plugin celu_plugin dump_tensor_plugin group_norm_plugin left_padding_cache_plugin \
mask_conv2d_sample_plugin rel_positional_encoding_plugin"

#echo $plugins

for p in $plugins
do
    `clang-format -i plugin/$p/*.h -style=file`
    `clang-format -i plugin/$p/*.cpp -style=file`
    `clang-format -i plugin/$p/*.cu -style=file`
    echo $p" clang-format done"
done

