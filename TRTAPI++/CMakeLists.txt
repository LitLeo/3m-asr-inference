# Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
include(cmake/modules/set_ifndef.cmake)
include(cmake/modules/find_library_create_target.cmake)

############################## Version ##################################
# The version number.
string(TIMESTAMP COMPILE_TIME "%Y%m%d")
set(TRT_PLUGIN_MAJOR 1)
set(TRT_PLUGIN_MINOR 0)
set(TRT_PLUGIN_PATCH 3)
set(TRT_PLUGIN_BUILD "Build${COMPILE_TIME}")
set(TRT_PLUGIN_VERSION "${TRT_PLUGIN_MAJOR}.${TRT_PLUGIN_MINOR}.${TRT_PLUGIN_PATCH}")

option(BUILD_GTESTS           "Build gtests" OFF)
option(BUILD_SELF_PLUGINS     "Build yourself plugins" ON)
option(BUILD_LIBTORCH_PLUGINS "Build plugins use libtorch" OFF)

set(PLUGIN_LISTS
    layer_norm_plugin
    glu_plugin
)
    #silu_plugin

# your self plugin
if (BUILD_SELF_PLUGINS)
    set(PLUGIN_LISTS 
        ${PLUGIN_LISTS}
        att_masked_softmax_plugin
        att_stream_softmax_plugin
        cat_split_cache_plugin
        dump_tensor_plugin
        fmoe_expert_plugin
        masked_fill_plugin
        mask_conv2d_sample_plugin
        rel_positional_encoding_plugin
    )
endif()
        #left_padding_cache_plugin

if (BUILD_LIBTORCH_PLUGINS)
    set(PLUGIN_LISTS 
        ${PLUGIN_LISTS}
        celu_plugin
        group_norm_plugin
    )
endif()

        #batch_norm_plugin

##########################################

set_ifndef(TRT_INSTALL_DIR "/data/TensorRT-7.2.2.3")
set(TRT_LIB_DIR "${TRT_INSTALL_DIR}/lib")
set(PROJECT_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/out")
message(${PROJECT_OUT_DIR})

set_ifndef(PROJECT_LIB_DIR ${CMAKE_BINARY_DIR})
set_ifndef(PROJECT_OUT_DIR ${CMAKE_BINARY_DIR})

file(STRINGS "${TRT_INSTALL_DIR}/include/NvInferVersion.h" VERSION_STRINGS REGEX "#define NV_TENSORRT_.*")

foreach(TYPE MAJOR MINOR PATCH BUILD)
    string(REGEX MATCH "NV_TENSORRT_${TYPE} [0-9]" TRT_TYPE_STRING ${VERSION_STRINGS})
    string(REGEX MATCH "[0-9]" TRT_${TYPE} ${TRT_TYPE_STRING})
endforeach(TYPE)

foreach(TYPE MAJOR MINOR PATCH)
    string(REGEX MATCH "NV_TENSORRT_SONAME_${TYPE} [0-9]" TRT_TYPE_STRING ${VERSION_STRINGS})
    string(REGEX MATCH "[0-9]" TRT_SO_${TYPE} ${TRT_TYPE_STRING})
endforeach(TYPE)

set(TRT_VERSION "${TRT_MAJOR}.${TRT_MINOR}.${TRT_PATCH}")
set(TRT_SOVERSION "${TRT_SO_MAJOR}")
#message("Building for TensorRT version: ${TRT_VERSION}, library version: ${TRT_SOVERSION}")

#if (TRT_MAJOR TRT_MAJOR "8")
#endif()

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++)
endif()

set(CMAKE_SKIP_BUILD_RPATH FALSE)

project(TRTAPI++
        LANGUAGES CXX CUDA
        VERSION ${TRT_PLUGIN_VERSION}
        DESCRIPTION "TensorRT is a C++ library that facilitates high performance inference on NVIDIA GPUs and deep learning accelerators."
        HOMEPAGE_URL "https://github.com/NVIDIA/TensorRT")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${TRT_LIB_DIR}/../ CACHE PATH "TensorRT installation" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations -D__SCORE_TIME__ -D__SCORE_GPU__ ${CMAKE_CXX_FLAGS}")

############################################################################################
# Cross-compilation settings

set_ifndef(TRT_PLATFORM_ID "x86_64")
message(STATUS "Targeting TRT Platform: ${TRT_PLATFORM_ID}")

############################################################################################
# Debug settings

set(TRT_DEBUG_POSTFIX _debug CACHE STRING "suffix for debug builds")

set_ifndef(CMAKE_BUILD_TYPE Debug)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Building in debug mode ${DEBUG_POSTFIX}")
    add_compile_options(-DBUILD_DEBUG)
endif()

if (BUILD_SELF_PLUGINS)
    add_compile_options(-DBUILD_SELF_PLUGINS)
endif()

############################################################################################
# Dependencies
set(DEFAULT_CUDA_VERSION 10.2)
set(DEFAULT_CUDNN_VERSION 8.0)
set(DEFAULT_CUB_VERSION 1.8.0)

# Dependency Version Resolution
set_ifndef(CUDA_VERSION ${DEFAULT_CUDA_VERSION})
message(STATUS "CUDA version set to ${CUDA_VERSION}")
set_ifndef(CUDNN_VERSION ${DEFAULT_CUDNN_VERSION})
message(STATUS "cuDNN version set to ${CUDNN_VERSION}")

find_package(Threads REQUIRED)
if (BUILD_PLUGINS OR BUILD_PARSERS)
    include(third_party/zlib.cmake)
    #include(third_party/protobuf.cmake)
endif()
if(NOT CUB_ROOT_DIR)
    set(CUB_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cub CACHE STRING "directory of CUB installation")
endif()

## find_package(CUDA) is broken for cross-compilation. Enable CUDA language instead.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    find_package(CUDA ${CUDA_VERSION} REQUIRED)
endif()

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${CUDNN_ROOT_DIR}/include
    ${TRT_INSTALL_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# cuda
find_library_create_target(cudnn cudnn STATIC ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library_create_target(cublas cublas STATIC ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library_create_target(cublasLt cublasLt STATIC ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library_create_target(cudart cudart STATIC ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
#find_library_create_target(culibos culibos STATIC ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

#set(CUDA_LIBRARIES ${CUDNN_LIB} ${CUBLAS_LIB} ${CUBLASLT_LIB} ${CUDART_LIB} ${CULIBOS_LIB} ${CUDART_LIB})
set(CUDA_LIBRARIES cudart cublas cublasLt cudnn)

find_library_create_target(nvinfer nvinfer SHARED ${TRT_LIB_DIR})

find_library(RT_LIB rt)

## torch
if(BUILD_GTESTS)
    set_ifndef(Torch_INSTALL_DIR "/usr/local/libtorch/" CACHE STRING "")
    set(Torch_DIR "${Torch_INSTALL_DIR}/share/cmake/Torch" CACHE STRING "Torch dir" FORCE)
    find_package(Torch REQUIRED)

    message(STATUS "Torch_VERSION = ${Torch_VERSION}") 
    message(STATUS "Torch_VERSION_MAJOR = ${Torch_VERSION_MAJOR}") 
    message(STATUS "Torch_VERSION_MINOR = ${Torch_VERSION_MINOR}") 
    message(STATUS "Torch_VERSION_PATCH = ${Torch_VERSION_PATCH}")    
else()
    set(Torch_VERSION "None")
endif()


if (BUILD_LIBTORCH_PLUGINS)
    add_compile_options(-DBUILD_LIBTORCH_PLUGINS)
else()
    message(WARNING "Not build plugins use libtorch, some plugins will not compile") 
endif()

############################################################################################
# CUDA targets

if (DEFINED GPU_ARCHS)
  message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
  separate_arguments(GPU_ARCHS)
else()
  list(APPEND GPU_ARCHS 61 70 75)

  string(REGEX MATCH "aarch64" IS_ARM "${TRT_PLATFORM_ID}")
  if (IS_ARM)
    # Xavier (SM72) only supported for aarch64.
    list(APPEND GPU_ARCHS 72)
  endif()

  if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
    # Ampere GPU (SM80) support is only available in CUDA versions > 11.0
    list(APPEND GPU_ARCHS 80)
    add_compile_definitions(THRUST_IGNORE_CUB_VERSION_CHECK)
  else()
    message(WARNING "Detected CUDA version is < 11.0. SM80 not supported.")
  endif()

  message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
endif()
set(BERT_GENCODES)
# Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
    if (${arch} GREATER_EQUAL 70)
        set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
    endif()
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
# Generate PTX for the last architecture in the list.
list(GET GPU_ARCHS -1 LATEST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
if (${LATEST_SM} GREATER_EQUAL 70)
    set(BERT_GENCODES "${BERT_GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")

############################################################################################
set(INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/TorchTrtPlugin${TRT_PLUGIN_VERSION}_cuda${CUDA_VERSION}_trt${TRT_VERSION}")

add_subdirectory(plugin)
add_subdirectory(python)

if(BUILD_GTESTS)
    add_subdirectory(gtests)
endif()


########################################  LOG  ###############################################
set(GPU_ARCHS_STR "${GPU_ARCHS}")
include(${PROJECT_SOURCE_DIR}/cmake/modules/utils.cmake)
message("=========================================================================")
string_status("Build Version"     ${TRT_PLUGIN_VERSION})
string_status("Build Time"        ${TRT_PLUGIN_BUILD})
string_status("Build Type"        ${CMAKE_BUILD_TYPE})
string_status("Build User"        $ENV{USER})
string_status("CUDA Version"      ${CUDA_VERSION})
string_status("CUDNN Version"     ${CUDNN_VERSION})
string_status("Torch Version"     ${Torch_VERSION})
string_status("TensorRT Version"  ${TRT_VERSION})
string_status("GPU Archs"         ${GPU_ARCHS_STR})
option_status("BUILD_GTESTS"      ${BUILD_GTESTS})
option_status("BUILD_SELF_PLUGINS"      ${BUILD_SELF_PLUGINS})
option_status("BUILD_LIBTORCH_PLUGINS"   ${BUILD_LIBTORCH_PLUGINS})
message("=========================================================================")

