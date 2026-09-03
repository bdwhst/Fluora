// CUDA registration TU for RhiTest's kernels (the primitives themselves are
// registered by the backend in rhi_cuda.cu).
#include "rhi_test_gpu.h"
#include "../rhi/rhi_cuda.h"

RHI_CUDA_REGISTER_KERNEL(test_tex_sample);
