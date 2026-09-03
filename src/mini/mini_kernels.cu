// CUDA compilation unit for the FluoraMini kernels: the single-source
// pathtrace_gpu.h compiled by nvcc (the Metal backend compiles the same file
// as MSL at runtime), plus one registration line per entry point so
// rhi::Device::createPipeline finds them by name (rhi_cuda.h). The wf_shade
// specializations mirror the rhi::SpecConstant values mini_main.cpp requests.
#include "pathtrace_gpu.h"
#include "../rhi/rhi_cuda.h"

RHI_CUDA_REGISTER_KERNEL(pathtraceKernel);
RHI_CUDA_REGISTER_KERNEL(wf_raygen);
RHI_CUDA_REGISTER_KERNEL(wf_prep_intersect);
RHI_CUDA_REGISTER_KERNEL(wf_prep_shade);
RHI_CUDA_REGISTER_KERNEL(wf_intersect);
RHI_CUDA_REGISTER_SPEC(wf_shade, 0, MINI_MAT_DIFFUSE);
RHI_CUDA_REGISTER_SPEC(wf_shade, 0, MINI_MAT_CONDUCTOR);
RHI_CUDA_REGISTER_SPEC(wf_shade, 0, MINI_MAT_GLASS);
RHI_CUDA_REGISTER_KERNEL(present_tonemap);
