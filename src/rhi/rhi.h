#pragma once
// =============================================================================
// RHI: the host-side seam between the renderer and a GPU backend. Two real
// implementations, one per platform, selected by kNativeBackend:
// rhi_metal.mm (macOS) and rhi_cuda.cu + rhi_cuda.h (Windows/CUDA). This
// header pins down the interface shape and the portability invariants each
// backend must satisfy.
//
// The two invariants that make a Metal backend possible at all:
//
//  1. No raw host pointers in device-visible data. Persistent cross-object
//     references are {tag, index} handles (taggedindex.h) resolved against a
//     TypedPoolView bound per dispatch. CUDA can cheat with unified memory;
//     Metal cannot (MTLBuffer.contents != gpuAddress).
//
//  2. Kernels are named entry points taking one parameter block (plain bytes)
//     plus resources referenced through it. No host-lambda launches
//     (GPUParallelFor's extended-lambda path is CUDA-only); each launch site
//     becomes a registered kernel. Metal: MTLComputePipelineState from a
//     metallib. CUDA: a launch thunk registered at static-init time.
// =============================================================================
#include <array>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace rhi {

enum class BackendKind { CUDA, Metal };

// The backend compiled into this binary (one per platform). Apps and tests
// create their device with this; a DeviceDesc::shaderSource is only needed
// for Metal (runtime MSL compile) — CUDA kernels come pre-registered.
#if defined(__APPLE__)
constexpr BackendKind kNativeBackend = BackendKind::Metal;
#else
constexpr BackendKind kNativeBackend = BackendKind::CUDA;
#endif
inline const char* backendName(BackendKind kind)
{
    return kind == BackendKind::Metal ? "metal" : "cuda";
}

// 64-bit GPU virtual address, embeddable in kernel parameter blocks.
// CUDA: the device pointer itself. Metal: MTLBuffer.gpuAddress (+offset),
// requires argument-buffer tier 2 — fine on all Apple Silicon.
using DeviceAddress = uint64_t;

struct Capabilities {
    BackendKind kind;
    bool hardwareRayTracing;   // Metal: MTLAccelerationStructure; CUDA: false (no OptiX path)
    bool unifiedMemory;        // both true in practice (managed memory / Apple Silicon)
};

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

enum class MemoryLocation {
    DeviceLocal,   // cudaMalloc            / MTLStorageModePrivate
    Shared,        // cudaMallocManaged     / MTLStorageModeShared
};

struct BufferDesc {
    size_t size = 0;
    MemoryLocation location = MemoryLocation::DeviceLocal;
    const char* debugName = nullptr;
};

class Buffer {
public:
    virtual ~Buffer() = default;
    virtual size_t size() const = 0;
    virtual DeviceAddress deviceAddress() const = 0;
    // Host-visible mapping; null for DeviceLocal. Valid to write only before
    // the buffer is in flight (Shared-mode coherency rules are the caller's
    // problem, same as today with managed memory).
    virtual void* hostPtr() = 0;
};

enum class TextureFormat { RGBA8Unorm, RGBA32Float, R32Float };

struct TextureDesc {
    int width = 0, height = 0;
    TextureFormat format = TextureFormat::RGBA8Unorm;
    bool normalizedCoords = true;  // cudaTextureDesc.normalizedCoords / sampler config
    bool srgb = false;
    const char* debugName = nullptr;
};

// Sampled image accessed through the device's bindless texture heap (see
// Device::textureHeap). shaderHandle() is the texture's index into that heap —
// an index, NOT a pointer, so it survives the host/device boundary (invariant
// 1) and can be stored in material structs / kernel params on both backends.
// Device code samples via the tex_heap_sample shim (src/rhi/texture.metal).
// Sampler state is fixed by the shim: bilinear + wrap, matching the
// cudaTextureDesc the CUDA renderer uses today.
class Texture {
public:
    virtual ~Texture() = default;
    virtual uint64_t shaderHandle() const = 0;
    virtual void upload(const void* pixels, size_t bytes) = 0;
};

// ---------------------------------------------------------------------------
// Compute
// ---------------------------------------------------------------------------

struct SpecConstant {
    uint32_t index;
    uint32_t value;
};

struct ComputePipelineDesc {
    // Name of a kernel entry point. CUDA: key into the kernel registry
    // (RHI_CUDA_REGISTER_KERNEL in rhi_cuda.h). Metal: function name in the library.
    std::string entryPoint;
    // Specialization constants baked at pipeline creation — one kernel source,
    // N specialized pipelines (e.g. per-material shade kernels). Metal:
    // [[function_constant(index)]] + MTLFunctionConstantValues. CUDA: template
    // instantiations registered under entryPoint + constant values
    // (RHI_CUDA_REGISTER_SPEC).
    std::vector<SpecConstant> constants;
};

class ComputePipeline {
public:
    virtual ~ComputePipeline() = default;
};

struct Dim3 { uint32_t x = 1, y = 1, z = 1; };

// Window/display size in window units (points on macOS, pixels elsewhere).
struct Extent2D { int width = 0, height = 0; };

// Ordered command recording + submission. CUDA: a cudaStream_t, dispatch =
// kernel launch. Metal: MTLCommandBuffer + compute encoders, submit = commit.
// The renderer's per-bounce loop (raygen -> intersect -> shade -> queues)
// records into one stream per frame.
//
// Binding convention: `params` is a POD blob copied at record time (Metal:
// setBytes at buffer(0); CUDA: part of the kernel argument struct). Resource
// i of `buffers` binds at Metal buffer(i+1); the CUDA thunk receives the
// corresponding device addresses in order. Grid semantics match CUDA:
// grid = number of thread groups, block = threads per group.
class CommandStream {
public:
    virtual ~CommandStream() = default;

    virtual void dispatch(ComputePipeline& pipeline, Dim3 grid, Dim3 block,
                          const void* params, size_t paramsSize,
                          std::initializer_list<Buffer*> buffers) = 0;

    // Indirect dispatch off a GPU-written count — required for the wavefront
    // work-queue design (queue sizes are only known on device). argsBuffer at
    // argsOffset holds three uint32 threadgroup counts {x, y, z}.
    // CUDA: a one-thread launcher kernel reads the counts on device and launches
    // the real grid via dynamic parallelism. Metal: dispatchThreadgroups(
    // indirectBuffer:).
    virtual void dispatchIndirect(ComputePipeline& pipeline, Dim3 block,
                                  Buffer& argsBuffer, size_t argsOffset,
                                  const void* params, size_t paramsSize,
                                  std::initializer_list<Buffer*> buffers) = 0;

    virtual void copy(Buffer& dst, size_t dstOffset,
                      const Buffer& src, size_t srcOffset, size_t bytes) = 0;
    virtual void fill(Buffer& dst, size_t offset, size_t bytes, uint8_t value) = 0;

    virtual void submit() = 0;
    virtual void waitIdle() = 0;   // cudaStreamSynchronize / waitUntilCompleted
};

// ---------------------------------------------------------------------------
// Ray tracing seam
// ---------------------------------------------------------------------------
//
// Host side: the CPU builds the acceleration structure (src/core/bvh_builder,
// backend-neutral); the intersector owns its GPU residency. Device side:
// kernels call rt_closest_hit() (src/rhi/raytrace_gpu.h), compiled per
// backend — the compute threaded-BVH traversal today, MSL intersection_query
// against an MTLAccelerationStructure as the M5 fast path (which would ignore
// `nodes` and build from the triangles instead). Kernel code cannot tell
// which one it got.

struct AccelBuildInput {
    const void* nodes = nullptr;      // RtBvhNode[numNodesPerDir * 6] (accel_shared.h)
    size_t nodeBytes = 0;
    uint32_t numNodesPerDir = 0;
    const void* triangles = nullptr;  // uint4 {i0,i1,i2,userData}
    size_t triangleBytes = 0;
    const void* positions = nullptr;  // float3 (16-byte stride)
    size_t positionBytes = 0;
};

class RayIntersector {
public:
    virtual ~RayIntersector() = default;
    virtual void build(const AccelBuildInput& input) = 0;
    virtual uint32_t numNodes() const = 0;
    // Buffers in the order rt_closest_hit expects (nodes, tris, positions),
    // for slot binding in dispatch(). An opaque bindless traversal view
    // replaces this when argument buffers land.
    virtual std::array<Buffer*, 3> bindings() = 0;
};

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

struct DeviceDesc {
    // Metal: MSL source compiled at device creation (newLibraryWithSource).
    // The host concatenates shared-struct headers + kernel files before passing
    // it in, since runtime MSL compilation cannot resolve #includes. Later
    // milestones replace this with a precompiled .metallib. CUDA: unused
    // (kernels are registered at static-init time).
    std::string shaderSource;
    // Compile shaderSource without fast math (Metal MTLMathModeSafe). Fast
    // math contracts/reassociates the same source differently per kernel, so
    // shared code can produce last-ulp-different values in the megakernel vs
    // the specialized wf_shade — this flag restores the mega == wavefront
    // bitwise invariant for verification runs, at ~40-50% mega / 0-12%
    // wavefront cost. CUDA: kernels are compiled offline, so this flag
    // cannot change codegen at runtime; its lowering is the
    // FLUORA_CUDA_SAFE_MATH CMake option (-fmad=false on the RHI targets).
    // nvcc never enables -use_fast_math, but its default -fmad=true still
    // lets ptxas contract shared expressions differently per kernel, so
    // requesting safeMath on a build without the option logs a warning.
    bool safeMath = false;
};

class Device {
public:
    virtual ~Device() = default;
    virtual Capabilities capabilities() const = 0;

    virtual std::unique_ptr<Buffer> createBuffer(const BufferDesc&) = 0;
    virtual std::unique_ptr<Texture> createTexture(const TextureDesc&) = 0;
    virtual std::unique_ptr<ComputePipeline> createPipeline(const ComputePipelineDesc&) = 0;
    virtual std::unique_ptr<CommandStream> createStream() = 0;
    virtual std::unique_ptr<RayIntersector> createIntersector() = 0;

    // Bindless texture table: one 64-bit entry per created texture, indexed by
    // Texture::shaderHandle(). Slot-bind it like any buffer and sample with
    // tex_heap_sample(heap, idx, uv). Metal: MTLResourceID entries (the backend
    // keeps every texture resident for dispatches automatically); CUDA: an
    // array of cudaTextureObject_t — same 8-byte layout, same kernel code.
    virtual Buffer& textureHeap() = 0;

    // Presentation seam replacing the CUDA<->OpenGL PBO interop in main.cpp:
    // the backend owns how a final image reaches the window (CUDA: GLFW window
    // + GL texture fed through an interop PBO, rhi_cuda_present.cpp; Metal:
    // Cocoa window + CAMetalLayer). Preview/ImGui code talks only to this.
    //
    // presentTarget() creates the window on first call and returns the RGBA8
    // buffer (width*height*4, row-major, row 0 = top of the window) a tonemap
    // kernel writes into. A later call with a different size resizes the
    // window to it and replaces the target (the previous Buffer& is dead; the
    // caller drains its streams first, as for any buffer it frees). The window
    // is never user-resizable: its size is the render size, set only through
    // this call. present() pumps window events and blits it to screen; it
    // returns false once the user asked to close (window close, q, Esc) —
    // after that the caller should stop rendering and exit. Writes to the
    // target must be submitted on this device's streams before present() so
    // same-queue ordering makes them visible.
    virtual Buffer& presentTarget(int width, int height) = 0;
    virtual bool present() = 0;

    // Largest window content size the primary display can show (its work area
    // minus the window frame), or {0, 0} when there is no display to ask. Since
    // the window size is the render size, a caller that lets scene data pick
    // the resolution uses this to keep the window on-screen.
    virtual Extent2D displaySize() const { return {}; }

    // Optional ImGui overlay. Enable before the first presentTarget(): the
    // backend creates the ImGui context and its platform/renderer backends and,
    // inside present(), builds a frame and invokes `draw` to issue ImGui::
    // widget calls, compositing them over the rendered image. The callback is
    // portable (issues ImGui:: calls only; lives in src/core/gui) and reads
    // input through ImGui's backend-neutral IO. No-op on backends built without
    // a GUI implementation.
    using GuiDrawFn = std::function<void()>;
    virtual void enableGui(const GuiDrawFn& /*draw*/) {}
};

std::unique_ptr<Device> createDevice(BackendKind kind, const DeviceDesc& desc = {});

} // namespace rhi
