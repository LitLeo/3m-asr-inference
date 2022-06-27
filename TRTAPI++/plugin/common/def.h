#ifndef PLUGIN_COMMON_DEF_H_
#define PLUGIN_COMMON_DEF_H_

#include "NvInferVersion.h"
#include "cuda.h"

#if NV_TENSORRT_MAJOR > 7
#undef TRTNOEXCEPT
#define TRTNOEXCEPT noexcept
#endif

#endif  // PLUGIN_COMMON_DEF_H_
