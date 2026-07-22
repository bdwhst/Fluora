#pragma once
// =============================================================================
// CUDA backend sketch for the RHI (rhi.h). Shows that the existing code maps
// onto the interface with thin wrappers — the point of writing it is to prove
// the interface is implementable without touching kernel internals. Not wired
// into the build; pathtrace.cu still calls CUDA directly.
//
// Migration map (existing code -> RHI):
//   cudaMalloc / cudaMallocManaged        -> Device::createBuffer(DeviceLocal/Shared)
//   cudaTextureObject_t + loadTexture...  -> Device::createTexture / Texture::shaderHandle
//   kernel<<<grid, block>>>(args...)      -> CommandStream::dispatch(pipeline, params blob)
//   GPUParallelFor(lambda)                -> named kernel + RHI_REGISTER_KERNEL
//   MTBVH build (bvh.cpp) + traversal     -> RayIntersector (build stays on CPU)
//   cudaGLMapBufferObject PBO dance       -> Device::presentTarget / present
// =============================================================================
#include "rhi.h"

#include <cuda_runtime.h>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace rhi {

inline void cudaCheck(cudaError_t err)
{
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}

class CudaBuffer final : public Buffer {
public:
    CudaBuffer(const BufferDesc& desc) : mSize(desc.size), mShared(desc.location == MemoryLocation::Shared)
    {
        if (mShared)
            cudaCheck(cudaMallocManaged(&mPtr, mSize, cudaMemAttachGlobal));
        else
            cudaCheck(cudaMalloc(&mPtr, mSize));
    }
    ~CudaBuffer() override { cudaFree(mPtr); }
    size_t size() const override { return mSize; }
    // CUDA's device address IS the pointer — unified virtual addressing.
    DeviceAddress deviceAddress() const override { return (DeviceAddress)(uintptr_t)mPtr; }
    void* hostPtr() override { return mShared ? mPtr : nullptr; }
private:
    void* mPtr = nullptr;
    size_t mSize;
    bool mShared;
};

// Kernel registry: every launch site becomes a named thunk. A thunk unpacks
// the parameter blob + slot-bound buffer addresses into the real kernel's
// argument struct and launches it. Per the rhi.h binding convention, thunks
// receive the device addresses of `buffers` in slot order.
//
//   struct ShadeParams { MaterialPoolView pool; int n; };  // buffer(0) blob
//   __global__ void shadeKernel(ShadeParams p, PathSegment* paths) { ... }
//   RHI_REGISTER_KERNEL: launch as kernel<<<>>>(params, (PathSegment*)addrs[0])
//
// Metal reaches the same kernel by compiling its MSL twin into the metallib
// under the same entry-point name.
using KernelThunk = std::function<void(Dim3 grid, Dim3 block, const void* params,
                                       size_t paramsSize,
                                       const std::vector<DeviceAddress>& bufferAddrs,
                                       cudaStream_t stream)>;

inline std::unordered_map<std::string, KernelThunk>& kernelRegistry()
{
    static std::unordered_map<std::string, KernelThunk> registry;
    return registry;
}

// SKETCH: parameter-only form; kernels taking slot-bound buffers get a
// variadic variant that casts bufferAddrs[i] to the i-th pointer parameter.
#define RHI_REGISTER_KERNEL(kernelFn, ParamsT)                                        \
    static const bool kernelFn##_registered = [] {                                    \
        rhi::kernelRegistry()[#kernelFn] = [](rhi::Dim3 g, rhi::Dim3 b,               \
                                              const void* p, size_t n,                \
                                              const std::vector<rhi::DeviceAddress>&, \
                                              cudaStream_t s) {                       \
            ParamsT params;                                                           \
            memcpy(&params, p, sizeof(ParamsT));                                      \
            kernelFn<<<dim3(g.x, g.y, g.z), dim3(b.x, b.y, b.z), 0, s>>>(params);     \
        };                                                                            \
        return true;                                                                  \
    }();

class CudaComputePipeline final : public ComputePipeline {
public:
    explicit CudaComputePipeline(const ComputePipelineDesc& desc)
        : thunk(kernelRegistry().at(desc.entryPoint)) {}
    KernelThunk thunk;
};

class CudaCommandStream final : public CommandStream {
public:
    CudaCommandStream() { cudaCheck(cudaStreamCreate(&mStream)); }
    ~CudaCommandStream() override { cudaStreamDestroy(mStream); }

    void dispatch(ComputePipeline& pipeline, Dim3 grid, Dim3 block,
                  const void* params, size_t paramsSize,
                  std::initializer_list<Buffer*> buffers) override
    {
        std::vector<DeviceAddress> addrs;
        for (Buffer* b : buffers)
            addrs.push_back(b->deviceAddress());
        static_cast<CudaComputePipeline&>(pipeline).thunk(grid, block, params, paramsSize, addrs, mStream);
    }

    void dispatchIndirect(ComputePipeline&, Dim3, Buffer&, size_t,
                          const void*, size_t, std::initializer_list<Buffer*>) override
    {
        // SKETCH: launch with worst-case grid + device-side early-out against
        // the count (what ForAllQueued does today), until a cuLaunch-based
        // indirect path is worth the trouble.
        throw std::logic_error("not implemented");
    }

    void copy(Buffer& dst, size_t dstOff, const Buffer& src, size_t srcOff, size_t bytes) override
    {
        cudaCheck(cudaMemcpyAsync((char*)(uintptr_t)dst.deviceAddress() + dstOff,
                                  (const char*)(uintptr_t)src.deviceAddress() + srcOff,
                                  bytes, cudaMemcpyDeviceToDevice, mStream));
    }
    void fill(Buffer& dst, size_t off, size_t bytes, uint8_t v) override
    {
        cudaCheck(cudaMemsetAsync((char*)(uintptr_t)dst.deviceAddress() + off, v, bytes, mStream));
    }

    void submit() override {}  // CUDA streams submit eagerly
    void waitIdle() override { cudaCheck(cudaStreamSynchronize(mStream)); }
private:
    cudaStream_t mStream;
};

// Traversal view the CUDA rt::intersect() unpacks — mirrors the fields of
// SceneInfoDev that intersect_surface_mtbvh reads today.
struct CudaTraversalView {
    DeviceAddress mtbvhArray;      // MTBVHGPUNode*, 6 direction-ordered arrays
    int bvhDataSize;
    DeviceAddress primitives;      // Primitive*
    DeviceAddress meshes;          // TriangleMesh*
};
static_assert(sizeof(CudaTraversalView) <= kTraversalViewMaxSize);

// SKETCH: CudaRayIntersector::build uploads the CPU-built MTBVH (bvh.cpp)
// exactly as pathtraceInit does now, then packs CudaTraversalView into the
// opaque blob. CudaTexture wraps the cudaArray/cudaTextureObject_t path that
// Scene::loadTextureFromFile implements. CudaDevice::presentTarget keeps the
// GL PBO interop currently in main.cpp/preview.cpp.

} // namespace rhi
