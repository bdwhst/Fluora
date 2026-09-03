// CUDA backend for the RHI (docs/metal-rhi-design.md, milestone M4).
// Implements Device / Buffer / Texture / ComputePipeline / CommandStream /
// RayIntersector over the CUDA runtime, and the presentation seam through
// rhi_cuda_present.cpp (GLFW + GL interop + ImGui). Kernels reach the
// backend through the registry in rhi_cuda.h; this file registers the
// parallel primitives (primitives_gpu.h) the Metal backend compiles from the
// same source, so rhi::Algorithms works unchanged.
//
// Memory model: DeviceLocal = cudaMalloc, Shared = cudaMallocManaged. On
// Windows (no concurrentManagedAccess) the host may touch managed memory only
// while no kernel is running on the device — the same "not in flight" rule
// rhi.h already imposes on Buffer::hostPtr(); the texture heap therefore
// lives in device memory with a host shadow instead of managed memory, so
// texture creation never races an in-flight dispatch.
#include "rhi_cuda.h"
#include "rhi_cuda_present.h"
#include "primitives_gpu.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace rhi {
namespace cuda {

// ---------------------------------------------------------------------------
// Kernel registry
// ---------------------------------------------------------------------------

namespace {

std::string entryKey(const std::string& name, const std::vector<SpecConstant>& constants)
{
    // Constants are sorted by index so {a,b} and {b,a} describe one pipeline.
    std::vector<SpecConstant> c = constants;
    std::sort(c.begin(), c.end(), [](const SpecConstant& x, const SpecConstant& y) {
        return x.index < y.index;
    });
    std::string key = name;
    for (const SpecConstant& sc : c)
        key += "|" + std::to_string(sc.index) + "=" + std::to_string(sc.value);
    return key;
}

std::unordered_map<std::string, KernelEntry>& registry()
{
    static std::unordered_map<std::string, KernelEntry> r;
    return r;
}

void cudaCheck(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

} // namespace

void registerEntry(KernelEntry entry)
{
    std::string key = entryKey(entry.name, entry.constants);
    registry()[key] = std::move(entry);
}

const KernelEntry* findEntry(const std::string& name, const std::vector<SpecConstant>& constants)
{
    auto it = registry().find(entryKey(name, constants));
    return it == registry().end() ? nullptr : &it->second;
}

// Primitives: same kernels the Metal library compiles from primitives_gpu.h.
RHI_CUDA_REGISTER_KERNEL(prim_reduce_sum);
RHI_CUDA_REGISTER_KERNEL(prim_scan_block);
RHI_CUDA_REGISTER_KERNEL(prim_scan_add_offsets);
RHI_CUDA_REGISTER_KERNEL(prim_compact_scatter);
RHI_CUDA_REGISTER_KERNEL(prim_radix_histogram);
RHI_CUDA_REGISTER_KERNEL(prim_radix_scatter);
RHI_CUDA_REGISTER_KERNEL(prim_queue_push_test);

} // namespace cuda

// ---------------------------------------------------------------------------
// Backend objects
// ---------------------------------------------------------------------------

class CudaDevice;
class CudaCommandStream;

class CudaBuffer final : public Buffer {
public:
    CudaBuffer(CudaDevice& dev, const BufferDesc& desc);
    ~CudaBuffer() override;
    size_t size() const override { return mSize; }
    DeviceAddress deviceAddress() const override { return (DeviceAddress)(uintptr_t)mPtr; }
    void* hostPtr() override { return mShared ? mPtr : nullptr; }
    void* ptr() const { return mPtr; }
private:
    CudaDevice& mDev;
    void* mPtr = nullptr;
    size_t mSize;
    bool mShared;
};

// Bindless texture: a cudaArray + texture object whose 8-byte handle sits in
// the device's heap slot shaderHandle(). Recycled slots keep repeated scene
// swaps from leaking toward kMaxTextures (same policy as the Metal heap).
class CudaTexture final : public Texture {
public:
    CudaTexture(CudaDevice& dev, const TextureDesc& desc);
    ~CudaTexture() override;
    uint64_t shaderHandle() const override { return mSlot; }
    void upload(const void* pixels, size_t bytes) override;
private:
    CudaDevice& mDev;
    cudaArray_t mArray = nullptr;
    cudaTextureObject_t mTex = 0;
    uint64_t mSlot = 0;
    int mWidth, mHeight;
    size_t mBytesPerPixel = 4;
};

class CudaComputePipeline final : public ComputePipeline {
public:
    explicit CudaComputePipeline(const ComputePipelineDesc& desc)
    {
        entry = cuda::findEntry(desc.entryPoint, desc.constants);
        if (!entry) {
            std::string what = "CUDA kernel not registered: " + desc.entryPoint;
            for (const SpecConstant& c : desc.constants)
                what += " [" + std::to_string(c.index) + "=" + std::to_string(c.value) + "]";
            throw std::runtime_error(what + " (RHI_CUDA_REGISTER_KERNEL/SPEC in a .cu file)");
        }
    }
    const cuda::KernelEntry* entry;
};

class CudaDevice final : public Device {
public:
    static constexpr size_t kMaxTextures = 1024;

    explicit CudaDevice(const DeviceDesc&)
    {
        cuda::cudaCheck(cudaSetDevice(0), "cudaSetDevice");
        cuda::cudaCheck(cudaFree(nullptr), "CUDA context creation");
        mHeapShadow.assign(kMaxTextures, 0);
        mHeap = std::make_unique<CudaBuffer>(
            *this, BufferDesc{ kMaxTextures * sizeof(cudaTextureObject_t),
                               MemoryLocation::DeviceLocal, "rhi.texheap" });
        cuda::cudaCheck(cudaMemset(mHeap->ptr(), 0, kMaxTextures * sizeof(cudaTextureObject_t)),
                        "texheap clear");
    }

    ~CudaDevice() override
    {
        cudaDeviceSynchronize();
        mPresenter.reset();
        mPresentBuf.reset();
        mHeap.reset();
        drainFrees(true);
    }

    Capabilities capabilities() const override
    {
        return Capabilities{ BackendKind::CUDA, false, true };
    }

    std::unique_ptr<Buffer> createBuffer(const BufferDesc& desc) override
    {
        drainFrees(false);
        return std::make_unique<CudaBuffer>(*this, desc);
    }

    std::unique_ptr<Texture> createTexture(const TextureDesc& desc) override
    {
        return std::make_unique<CudaTexture>(*this, desc);
    }

    std::unique_ptr<ComputePipeline> createPipeline(const ComputePipelineDesc& desc) override
    {
        return std::make_unique<CudaComputePipeline>(desc);
    }

    std::unique_ptr<CommandStream> createStream() override;
    std::unique_ptr<RayIntersector> createIntersector() override;

    Buffer& textureHeap() override { return *mHeap; }

    Buffer& presentTarget(int width, int height) override
    {
        if (!mPresenter) {
            mPresenter = std::make_unique<cuda::Presenter>(width, height,
                                                           mGuiEnabled ? &mGuiDraw : nullptr);
            mPresentBuf = std::make_unique<CudaBuffer>(
                *this, BufferDesc{ (size_t)width * height * 4, MemoryLocation::DeviceLocal,
                                   "rhi.present" });
            mPresentW = width;
            mPresentH = height;
        } else if (width != mPresentW || height != mPresentH) {
            throw std::logic_error("present target resize not supported");
        }
        return *mPresentBuf;
    }

    void enableGui(const GuiDrawFn& draw) override
    {
#ifdef RHI_ENABLE_IMGUI
        mGuiDraw = draw;
        mGuiEnabled = true;
#else
        (void)draw;
#endif
    }

    bool present() override
    {
        if (!mPresenter)
            throw std::logic_error("present() before presentTarget()");
        return mPresenter->present(mPresentBuf->ptr());
    }

    // ---- texture heap slots ----
    uint64_t acquireSlot(cudaTextureObject_t tex)
    {
        uint64_t slot;
        if (!mFreeSlots.empty()) {
            slot = mFreeSlots.back();
            mFreeSlots.pop_back();
        } else {
            slot = mHighWater++;
            if (slot >= kMaxTextures)
                throw std::runtime_error("texture heap full");
        }
        mHeapShadow[slot] = tex;
        // Synchronous H2D copy on the legacy stream: ordered after any
        // blocking-stream dispatch still reading the heap.
        cuda::cudaCheck(cudaMemcpy((char*)mHeap->ptr() + slot * sizeof(cudaTextureObject_t),
                                   &mHeapShadow[slot], sizeof(cudaTextureObject_t),
                                   cudaMemcpyHostToDevice),
                        "texheap write");
        return slot;
    }
    void releaseSlot(uint64_t slot)
    {
        mHeapShadow[slot] = 0;
        cudaMemcpy((char*)mHeap->ptr() + slot * sizeof(cudaTextureObject_t), &mHeapShadow[slot],
                   sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
        mFreeSlots.push_back(slot);
    }

    // ---- deferred frees (rhi_algorithms.cpp lifetime note) ----
    // A buffer may die host-side while dispatches that read it are still
    // queued. The free is deferred until an event recorded on every live
    // stream at release time has completed. (cudaFree also device-syncs
    // implicitly, so this is about not stalling, not about correctness.)
    void registerStream(cudaStream_t s) { mStreams.push_back(s); }
    void unregisterStream(cudaStream_t s)
    {
        mStreams.erase(std::remove(mStreams.begin(), mStreams.end(), s), mStreams.end());
    }
    void deferFree(void* ptr)
    {
        if (!ptr)
            return;
        PendingFree pf;
        pf.ptr = ptr;
        for (cudaStream_t s : mStreams) {
            cudaEvent_t ev;
            if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess)
                continue;
            if (cudaEventRecord(ev, s) != cudaSuccess) {
                cudaEventDestroy(ev);
                continue;
            }
            pf.events.push_back(ev);
        }
        mPending.push_back(std::move(pf));
    }
    void drainFrees(bool force)
    {
        for (auto it = mPending.begin(); it != mPending.end();) {
            bool done = true;
            for (cudaEvent_t ev : it->events) {
                cudaError_t st = force ? cudaEventSynchronize(ev) : cudaEventQuery(ev);
                if (st == cudaErrorNotReady) {
                    done = false;
                    break;
                }
            }
            if (!done) {
                ++it;
                continue;
            }
            for (cudaEvent_t ev : it->events)
                cudaEventDestroy(ev);
            cudaFree(it->ptr);
            it = mPending.erase(it);
        }
    }

private:
    struct PendingFree {
        void* ptr = nullptr;
        std::vector<cudaEvent_t> events;
    };

    std::unique_ptr<CudaBuffer> mHeap;
    std::vector<cudaTextureObject_t> mHeapShadow;
    std::vector<uint64_t> mFreeSlots;
    uint64_t mHighWater = 0;

    std::vector<cudaStream_t> mStreams;
    std::deque<PendingFree> mPending;

    std::unique_ptr<cuda::Presenter> mPresenter;
    std::unique_ptr<CudaBuffer> mPresentBuf;
    int mPresentW = 0, mPresentH = 0;
    bool mGuiEnabled = false;
    GuiDrawFn mGuiDraw;
};

// ---- CudaBuffer ----

CudaBuffer::CudaBuffer(CudaDevice& dev, const BufferDesc& desc)
    : mDev(dev), mSize(desc.size), mShared(desc.location == MemoryLocation::Shared)
{
    // Never hand out a null pointer for an empty buffer (Metal rejects
    // zero-length buffers; callers already pad to >= 16 bytes, mirror that).
    size_t bytes = std::max<size_t>(mSize, 16);
    if (mShared)
        cuda::cudaCheck(cudaMallocManaged(&mPtr, bytes, cudaMemAttachGlobal),
                        desc.debugName ? desc.debugName : "cudaMallocManaged");
    else
        cuda::cudaCheck(cudaMalloc(&mPtr, bytes), desc.debugName ? desc.debugName : "cudaMalloc");
}

CudaBuffer::~CudaBuffer()
{
    mDev.deferFree(mPtr);
}

// ---- CudaTexture ----

CudaTexture::CudaTexture(CudaDevice& dev, const TextureDesc& desc)
    : mDev(dev), mWidth(desc.width), mHeight(desc.height)
{
    cudaChannelFormatDesc cd;
    cudaTextureReadMode readMode;
    switch (desc.format) {
    case TextureFormat::RGBA8Unorm:
        cd = cudaCreateChannelDesc<uchar4>();
        readMode = cudaReadModeNormalizedFloat;
        mBytesPerPixel = 4;
        break;
    case TextureFormat::RGBA32Float:
        cd = cudaCreateChannelDesc<float4>();
        readMode = cudaReadModeElementType;
        mBytesPerPixel = 16;
        break;
    case TextureFormat::R32Float:
        cd = cudaCreateChannelDesc<float>();
        readMode = cudaReadModeElementType;
        mBytesPerPixel = 4;
        break;
    default:
        throw std::logic_error("unsupported texture format");
    }
    cuda::cudaCheck(cudaMallocArray(&mArray, &cd, desc.width, desc.height),
                    desc.debugName ? desc.debugName : "cudaMallocArray");

    cudaResourceDesc rd = {};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = mArray;
    // Fixed sampler state shared with texture_gpu.h's MSL sampler: bilinear,
    // wrap, normalized coordinates; sRGB decode on sample for 8-bit textures.
    cudaTextureDesc td = {};
    td.addressMode[0] = cudaAddressModeWrap;
    td.addressMode[1] = cudaAddressModeWrap;
    td.filterMode = cudaFilterModeLinear;
    td.readMode = readMode;
    td.normalizedCoords = desc.normalizedCoords ? 1 : 0;
    td.sRGB = (desc.srgb && desc.format == TextureFormat::RGBA8Unorm) ? 1 : 0;
    cuda::cudaCheck(cudaCreateTextureObject(&mTex, &rd, &td, nullptr), "cudaCreateTextureObject");
    mSlot = mDev.acquireSlot(mTex);
}

CudaTexture::~CudaTexture()
{
    mDev.releaseSlot(mSlot);
    cudaDestroyTextureObject(mTex);
    cudaFreeArray(mArray);
}

void CudaTexture::upload(const void* pixels, size_t bytes)
{
    size_t expected = (size_t)mWidth * mHeight * mBytesPerPixel;
    if (bytes != expected)
        throw std::logic_error("texture upload size mismatch");
    size_t pitch = (size_t)mWidth * mBytesPerPixel;
    cuda::cudaCheck(cudaMemcpy2DToArray(mArray, 0, 0, pixels, pitch, pitch, mHeight,
                                        cudaMemcpyHostToDevice),
                    "texture upload");
}

// ---- CudaCommandStream ----

class CudaCommandStream final : public CommandStream {
public:
    explicit CudaCommandStream(CudaDevice& dev) : mDev(dev)
    {
        // A blocking stream (default flags): it serializes with the legacy
        // default stream, which is what present() and texture-heap writes use,
        // giving present-after-submit ordering for free (Metal: one queue).
        cuda::cudaCheck(cudaStreamCreate(&mStream), "cudaStreamCreate");
        mDev.registerStream(mStream);
    }
    ~CudaCommandStream() override
    {
        cudaStreamSynchronize(mStream);
        mDev.unregisterStream(mStream);
        for (cudaEvent_t ev : mInFlight)
            cudaEventDestroy(ev);
        cudaStreamDestroy(mStream);
    }

    void dispatch(ComputePipeline& pipeline, Dim3 grid, Dim3 block,
                  const void* params, size_t paramsSize,
                  std::initializer_list<Buffer*> buffers) override
    {
        const cuda::KernelEntry* e = check(pipeline, params, paramsSize, buffers);
        if (grid.x == 0 || grid.y == 0 || grid.z == 0)
            return;
        e->launch(dim3(grid.x, grid.y, grid.z), dim3(block.x, block.y, block.z), params,
                  addresses(buffers), mStream);
        cuda::cudaCheck(cudaGetLastError(), e->name.c_str());
    }

    void dispatchIndirect(ComputePipeline& pipeline, Dim3 block,
                          Buffer& argsBuffer, size_t argsOffset,
                          const void* params, size_t paramsSize,
                          std::initializer_list<Buffer*> buffers) override
    {
        const cuda::KernelEntry* e = check(pipeline, params, paramsSize, buffers);
        const unsigned* args =
            (const unsigned*)((const char*)(uintptr_t)argsBuffer.deviceAddress() + argsOffset);
        e->launchIndirect(args, dim3(block.x, block.y, block.z), params, addresses(buffers),
                          mStream);
        cuda::cudaCheck(cudaGetLastError(), e->name.c_str());
    }

    void copy(Buffer& dst, size_t dstOffset,
              const Buffer& src, size_t srcOffset, size_t bytes) override
    {
        cuda::cudaCheck(cudaMemcpyAsync((char*)(uintptr_t)dst.deviceAddress() + dstOffset,
                                        (const char*)(uintptr_t)src.deviceAddress() + srcOffset,
                                        bytes, cudaMemcpyDeviceToDevice, mStream),
                        "copy");
    }

    void fill(Buffer& dst, size_t offset, size_t bytes, uint8_t value) override
    {
        cuda::cudaCheck(cudaMemsetAsync((char*)(uintptr_t)dst.deviceAddress() + offset, value,
                                        bytes, mStream),
                        "fill");
    }

    // Bounded in-flight submits, as in the Metal backend: an event marks each
    // submit and the CPU blocks once more than kMaxInFlight are pending, so
    // preview pacing tracks GPU progress instead of the CPU running seconds
    // ahead through the driver queue.
    static constexpr size_t kMaxInFlight = 2;

    void submit() override
    {
        cudaEvent_t ev;
        cuda::cudaCheck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        cuda::cudaCheck(cudaEventRecord(ev, mStream), "cudaEventRecord");
        mInFlight.push_back(ev);
        while (mInFlight.size() > kMaxInFlight) {
            cuda::cudaCheck(cudaEventSynchronize(mInFlight.front()), "GPU execution");
            cudaEventDestroy(mInFlight.front());
            mInFlight.pop_front();
        }
        mDev.drainFrees(false);
    }

    void waitIdle() override
    {
        cuda::cudaCheck(cudaStreamSynchronize(mStream), "GPU execution");
        for (cudaEvent_t ev : mInFlight)
            cudaEventDestroy(ev);
        mInFlight.clear();
        mDev.drainFrees(false);
    }

private:
    static const cuda::KernelEntry* check(ComputePipeline& pipeline, const void* params,
                                          size_t paramsSize,
                                          std::initializer_list<Buffer*> buffers)
    {
        const cuda::KernelEntry* e = static_cast<CudaComputePipeline&>(pipeline).entry;
        if (params && paramsSize != e->paramsSize)
            throw std::logic_error(e->name + ": params size " + std::to_string(paramsSize)
                                   + " != kernel parameter block " + std::to_string(e->paramsSize));
        if (buffers.size() < e->numBuffers)
            throw std::logic_error(e->name + ": " + std::to_string(buffers.size())
                                   + " buffers bound, kernel declares "
                                   + std::to_string(e->numBuffers));
        return e;
    }

    static std::vector<void*> addresses(std::initializer_list<Buffer*> buffers)
    {
        std::vector<void*> v;
        v.reserve(buffers.size());
        for (Buffer* b : buffers)
            v.push_back((void*)(uintptr_t)b->deviceAddress());
        return v;
    }

    CudaDevice& mDev;
    cudaStream_t mStream = nullptr;
    std::deque<cudaEvent_t> mInFlight;
};

std::unique_ptr<CommandStream> CudaDevice::createStream()
{
    return std::make_unique<CudaCommandStream>(*this);
}

// ---- CudaRayIntersector ----
// Compute-traversal intersector: uploads the CPU-built threaded BVH
// (core/bvh_builder); the paired device code is rt_closest_hit in
// raytrace_gpu.h — the same traversal the Metal backend runs.
class CudaRayIntersector final : public RayIntersector {
public:
    explicit CudaRayIntersector(CudaDevice& dev) : mDev(dev) {}

    void build(const AccelBuildInput& in) override
    {
        mNumNodes = in.numNodesPerDir;
        mNodes = upload(in.nodes, in.nodeBytes, "rt.nodes");
        mTris = upload(in.triangles, in.triangleBytes, "rt.tris");
        mPositions = upload(in.positions, in.positionBytes, "rt.positions");
    }
    uint32_t numNodes() const override { return mNumNodes; }
    std::array<Buffer*, 3> bindings() override
    {
        return { mNodes.get(), mTris.get(), mPositions.get() };
    }

private:
    std::unique_ptr<Buffer> upload(const void* data, size_t bytes, const char* name)
    {
        auto buf = std::make_unique<CudaBuffer>(
            mDev, BufferDesc{ std::max<size_t>(bytes, 16), MemoryLocation::DeviceLocal, name });
        if (data && bytes)
            cuda::cudaCheck(cudaMemcpy(buf->ptr(), data, bytes, cudaMemcpyHostToDevice), name);
        return buf;
    }
    CudaDevice& mDev;
    uint32_t mNumNodes = 0;
    std::unique_ptr<Buffer> mNodes, mTris, mPositions;
};

std::unique_ptr<RayIntersector> CudaDevice::createIntersector()
{
    return std::make_unique<CudaRayIntersector>(*this);
}

std::unique_ptr<Device> createDevice(BackendKind kind, const DeviceDesc& desc)
{
    if (kind != BackendKind::CUDA)
        throw std::logic_error("only the CUDA backend is available on this platform");
    return std::make_unique<CudaDevice>(desc);
}

} // namespace rhi
