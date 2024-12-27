#pragma once

#ifdef __CUDA_ARCH__
#define GPU_UNROLL #pragma unroll
#else
#define GPU_UNROLL
#endif

#define CPU_GPU_FUNC __host__ __device__ 
#define GPU_FUNC __device__