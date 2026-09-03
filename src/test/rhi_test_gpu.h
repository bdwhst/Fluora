#ifndef TEST_RHI_TEST_GPU_H
#define TEST_RHI_TEST_GPU_H
// RhiTest's own kernel(s), single-source like the renderer's: concatenated
// into the MSL library after texture_gpu.h, #included by rhi_test_kernels.cu
// for CUDA.

#ifndef __METAL_VERSION__
#include "../rhi/gpu_portable.h"
#include "../rhi/primitives_shared.h"
#include "../rhi/texture_gpu.h"
#endif

// Samples the bindless heap at (q.xy) from texture index q.z.
GPU_KERNEL(test_tex_sample, GPU_TID_1D)(GPU_KERNEL_PARAMS(PrimParams, P),
    GPU_BUFFER(const RhiTex, heap),
    GPU_BUFFER(const gpu_float4, q),
    GPU_BUFFER(gpu_float4, outv))
{
    uint tid = GPU_GLOBAL_ID_X;
    if (tid >= P.n)
        return;
    gpu_float4 qq = q[tid];
    outv[tid] = tex_heap_sample(heap, (uint)qq.z, gpu_float2(qq.x, qq.y));
}

#endif
