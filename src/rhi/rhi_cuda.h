#pragma once
// =============================================================================
// CUDA backend for the RHI (rhi.h) — kernel registration API.
//
// The backend proper (Device/Buffer/Texture/CommandStream/RayIntersector/
// present) lives in rhi_cuda.cu. This header is what a .cu file includes to
// make its kernels reachable by name (rhi::ComputePipelineDesc::entryPoint),
// mirroring the Metal backend compiling the same single-source kernels into a
// library under the same names:
//
//   #include "pathtrace_gpu.h"             // GPU_KERNEL(wf_intersect)(...)
//   #include "../rhi/rhi_cuda.h"
//   RHI_CUDA_REGISTER_KERNEL(wf_intersect);
//   RHI_CUDA_REGISTER_SPEC(wf_shade, 0, MINI_MAT_DIFFUSE);   // wf_shade<...>
//
// Binding convention (rhi.h): the params blob is copied into the kernel's
// first (by-value) parameter; resource i of the dispatch's buffer list becomes
// pointer parameter i+1 — exactly the GPU_KERNEL_PARAMS / GPU_BUFFER order the
// kernel macros declare, so the launch thunk is deduced from the kernel's own
// signature. No per-kernel argument struct, no lambda across the launch.
//
// Indirect dispatch (queue counts written on device) is the CUDA analog of
// dispatchThreadgroups(indirectBuffer:): a one-thread launcher kernel reads
// the {x,y,z} group counts and launches the real kernel with dynamic
// parallelism. The launcher is generated per registered kernel by the macro
// (a plain, uniquely named __global__): nvcc does not register __global__
// templates whose template argument is a kernel address, so the kernel
// identity lives in a __device__ helper's template argument instead. Needs
// relocatable device code + cudadevrt (set on every RHI target in
// CMakeLists.txt). The parent grid completes only after its child grid does,
// so stream order after the launcher is preserved.
// =============================================================================
#include "rhi.h"

#include <cuda_runtime.h>

#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rhi {
namespace cuda {

struct KernelEntry {
    std::string name;
    std::vector<SpecConstant> constants;   // empty for unspecialized kernels
    size_t paramsSize = 0;
    size_t numBuffers = 0;
    // Direct launch: grid/block from the host.
    std::function<void(dim3 grid, dim3 block, const void* params,
                       const std::vector<void*>& buffers, cudaStream_t stream)> launch;
    // Indirect launch: group counts read on device from `args` (3 x uint32).
    std::function<void(const unsigned* args, dim3 block, const void* params,
                       const std::vector<void*>& buffers, cudaStream_t stream)> launchIndirect;
};

// Global registry, populated by static initializers in each registering TU.
void registerEntry(KernelEntry entry);
const KernelEntry* findEntry(const std::string& name, const std::vector<SpecConstant>& constants);

// Signature introspection for a kernel `void k(P, B0*, B1*, ...)`: its
// parameter block type, buffer count, and a flat argument pack that crosses
// the launcher-kernel boundary as one by-value parameter.
template <class F>
struct KernelSig;

template <class P, class... Bs>
struct KernelSig<void (*)(P, Bs*...)> {
    using Params = P;
    static constexpr size_t NumBuffers = sizeof...(Bs);
    struct Pack {
        P p;
        void* bufs[NumBuffers > 0 ? NumBuffers : 1];
    };

    template <auto Kern, size_t... Is>
    __device__ static void launchDevice(dim3 grid, dim3 block, const Pack& pack,
                                        std::index_sequence<Is...>)
    {
        Kern<<<grid, block, 0, cudaStreamFireAndForget>>>(pack.p, static_cast<Bs*>(pack.bufs[Is])...);
    }

    template <auto Kern, size_t... Is>
    static void launchHost(dim3 grid, dim3 block, const Pack& pack, cudaStream_t stream,
                           std::index_sequence<Is...>)
    {
        Kern<<<grid, block, 0, stream>>>(pack.p, static_cast<Bs*>(pack.bufs[Is])...);
    }

    static Pack makePack(const void* params, const std::vector<void*>& bufs)
    {
        Pack pack;
        std::memset(&pack, 0, sizeof(Pack));
        if (params)
            std::memcpy(&pack.p, params, sizeof(P));
        for (size_t i = 0; i < NumBuffers; i++)
            pack.bufs[i] = bufs[i];
        return pack;
    }
};

// Body of every generated launcher kernel: one thread, reads the group counts,
// launches the child grid. A zero in any dimension skips the launch (CUDA
// rejects empty grids; Metal treats them as no-ops).
template <auto Kern>
__device__ inline void deviceLaunch(const unsigned* args, dim3 block,
                                    const typename KernelSig<decltype(Kern)>::Pack& pack)
{
    using Sig = KernelSig<decltype(Kern)>;
    unsigned gx = args[0], gy = args[1], gz = args[2];
    if (gx == 0 || gy == 0 || gz == 0)
        return;
    Sig::template launchDevice<Kern>(dim3(gx, gy, gz), block, pack,
                                     std::make_index_sequence<Sig::NumBuffers>{});
}

template <auto Kern>
bool registerKernel(const char* name, std::vector<SpecConstant> constants,
                    void (*launcher)(const unsigned*, dim3,
                                     typename KernelSig<decltype(Kern)>::Pack))
{
    using Sig = KernelSig<decltype(Kern)>;
    using Pack = typename Sig::Pack;
    KernelEntry e;
    e.name = name;
    e.constants = std::move(constants);
    e.paramsSize = sizeof(typename Sig::Params);
    e.numBuffers = Sig::NumBuffers;
    e.launch = [](dim3 grid, dim3 block, const void* params, const std::vector<void*>& bufs,
                  cudaStream_t stream) {
        Pack pack = Sig::makePack(params, bufs);
        Sig::template launchHost<Kern>(grid, block, pack, stream,
                                       std::make_index_sequence<Sig::NumBuffers>{});
    };
    e.launchIndirect = [launcher](const unsigned* args, dim3 block, const void* params,
                                  const std::vector<void*>& bufs, cudaStream_t stream) {
        Pack pack = Sig::makePack(params, bufs);
        void* kargs[] = { (void*)&args, (void*)&block, (void*)&pack };
        cudaLaunchKernel((const void*)launcher, dim3(1, 1, 1), dim3(1, 1, 1), kargs, 0, stream);
    };
    registerEntry(std::move(e));
    return true;
}

} // namespace cuda
} // namespace rhi

// kernel: the kernel expression (name, or name<value> for a specialization);
// tag: an identifier unique per registration; name/constants: the pipeline key.
#define RHI_CUDA_REGISTER_IMPL(kernel, tag, name, constants)                                  \
    __global__ void rhi_cuda_launch_##tag(const unsigned* args, dim3 block,                   \
                                          rhi::cuda::KernelSig<decltype(&kernel)>::Pack pack) \
    {                                                                                         \
        rhi::cuda::deviceLaunch<kernel>(args, block, pack);                                   \
    }                                                                                         \
    static const bool rhi_cuda_reg_##tag =                                                    \
        rhi::cuda::registerKernel<kernel>(name, constants, rhi_cuda_launch_##tag)

#define RHI_CUDA_REGISTER_KERNEL(kernel) \
    RHI_CUDA_REGISTER_IMPL(kernel, kernel, #kernel, {})

// Registers the template instantiation `kernel<value>` under entry point
// #kernel with specialization constant {index, value} (GPU_SPEC_CONST lowering
// in gpu_portable.h).
#define RHI_CUDA_REGISTER_SPEC(kernel, index, value)                                          \
    RHI_CUDA_REGISTER_IMPL(kernel<value>, kernel##_##index##_##value, #kernel,                \
                           (std::vector<rhi::SpecConstant>{ { (index), (value) } }))
