#pragma once

#ifdef __CUDA_ARCH__
#define GPU_UNROLL #pragma unroll
#else
#define GPU_UNROLL
#endif