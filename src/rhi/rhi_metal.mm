// Metal backend for the RHI (see docs/metal-rhi-design.md, milestone M1).
// Implements Device / Buffer / ComputePipeline / CommandStream /
// RayIntersector and the presentation seam (Cocoa window + CAMetalLayer).
// Texture is M3 work and throws for now.
// Compiled with ARC (-fobjc-arc); ObjC objects held as C++ members are strong.
#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "rhi.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>

// Flags the close button instead of tearing the window down: present() reports
// it and the app exits, which is also what q/Esc do.
@interface RhiWindowDelegate : NSObject <NSWindowDelegate>
@property(nonatomic) BOOL closeRequested;
@end
@implementation RhiWindowDelegate
- (BOOL)windowShouldClose:(NSWindow*)sender
{
    self.closeRequested = YES;
    return NO;
}
@end

namespace rhi {

static std::string nsErrorToString(NSError* err)
{
    return err ? std::string([[err localizedDescription] UTF8String]) : std::string("unknown error");
}

class MetalBuffer final : public Buffer {
public:
    MetalBuffer(id<MTLDevice> dev, const BufferDesc& desc)
    {
        MTLResourceOptions opts = (desc.location == MemoryLocation::Shared)
            ? MTLResourceStorageModeShared
            : MTLResourceStorageModePrivate;
        mBuf = [dev newBufferWithLength:desc.size options:opts];
        if (!mBuf)
            throw std::runtime_error("MTLBuffer allocation failed");
        if (desc.debugName)
            mBuf.label = [NSString stringWithUTF8String:desc.debugName];
    }
    size_t size() const override { return [mBuf length]; }
    DeviceAddress deviceAddress() const override { return [mBuf gpuAddress]; }
    void* hostPtr() override
    {
        return [mBuf storageMode] == MTLStorageModeShared ? [mBuf contents] : nullptr;
    }
    id<MTLBuffer> handle() const { return mBuf; }
private:
    id<MTLBuffer> mBuf;
};

class MetalComputePipeline final : public ComputePipeline {
public:
    MetalComputePipeline(id<MTLDevice> dev, id<MTLLibrary> lib, const ComputePipelineDesc& desc)
    {
        NSString* name = [NSString stringWithUTF8String:desc.entryPoint.c_str()];
        NSError* err = nil;
        id<MTLFunction> fn;
        if (desc.constants.empty()) {
            fn = [lib newFunctionWithName:name];
        } else {
            MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
            for (const SpecConstant& c : desc.constants)
                [values setConstantValue:&c.value type:MTLDataTypeUInt atIndex:c.index];
            fn = [lib newFunctionWithName:name constantValues:values error:&err];
        }
        if (!fn)
            throw std::runtime_error("Metal kernel not found: " + desc.entryPoint + ": " + nsErrorToString(err));
        mPso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!mPso)
            throw std::runtime_error("Pipeline creation failed for " + desc.entryPoint + ": " + nsErrorToString(err));
    }
    id<MTLComputePipelineState> pso() const { return mPso; }
private:
    id<MTLComputePipelineState> mPso;
};

class MetalCommandStream final : public CommandStream {
public:
    explicit MetalCommandStream(id<MTLCommandQueue> queue) : mQueue(queue) {}

    void dispatch(ComputePipeline& pipeline, Dim3 grid, Dim3 block,
                  const void* params, size_t paramsSize,
                  std::initializer_list<Buffer*> buffers) override
    {
        id<MTLComputeCommandEncoder> enc = [current() computeCommandEncoder];
        [enc setComputePipelineState:static_cast<MetalComputePipeline&>(pipeline).pso()];
        if (params && paramsSize)
            [enc setBytes:params length:paramsSize atIndex:0];
        NSUInteger slot = 1;
        for (Buffer* b : buffers)
            [enc setBuffer:static_cast<MetalBuffer*>(b)->handle() offset:0 atIndex:slot++];
        [enc dispatchThreadgroups:MTLSizeMake(grid.x, grid.y, grid.z)
            threadsPerThreadgroup:MTLSizeMake(block.x, block.y, block.z)];
        [enc endEncoding];
    }

    void dispatchIndirect(ComputePipeline& pipeline, Dim3 block,
                          Buffer& argsBuffer, size_t argsOffset,
                          const void* params, size_t paramsSize,
                          std::initializer_list<Buffer*> buffers) override
    {
        id<MTLComputeCommandEncoder> enc = [current() computeCommandEncoder];
        [enc setComputePipelineState:static_cast<MetalComputePipeline&>(pipeline).pso()];
        if (params && paramsSize)
            [enc setBytes:params length:paramsSize atIndex:0];
        NSUInteger slot = 1;
        for (Buffer* b : buffers)
            [enc setBuffer:static_cast<MetalBuffer*>(b)->handle() offset:0 atIndex:slot++];
        [enc dispatchThreadgroupsWithIndirectBuffer:static_cast<MetalBuffer&>(argsBuffer).handle()
                               indirectBufferOffset:argsOffset
                              threadsPerThreadgroup:MTLSizeMake(block.x, block.y, block.z)];
        [enc endEncoding];
    }

    void copy(Buffer& dst, size_t dstOffset,
              const Buffer& src, size_t srcOffset, size_t bytes) override
    {
        id<MTLBlitCommandEncoder> blit = [current() blitCommandEncoder];
        [blit copyFromBuffer:static_cast<const MetalBuffer&>(src).handle()
                sourceOffset:srcOffset
                    toBuffer:static_cast<MetalBuffer&>(dst).handle()
           destinationOffset:dstOffset
                        size:bytes];
        [blit endEncoding];
    }

    void fill(Buffer& dst, size_t offset, size_t bytes, uint8_t value) override
    {
        id<MTLBlitCommandEncoder> blit = [current() blitCommandEncoder];
        [blit fillBuffer:static_cast<MetalBuffer&>(dst).handle()
                   range:NSMakeRange(offset, bytes)
                   value:value];
        [blit endEncoding];
    }

    // Bounded in-flight submits: without backpressure the CPU encodes an
    // entire render ahead of the GPU, queueing seconds of GPU work that
    // starves WindowServer compositing (OS-wide stutter on heavy scenes) and
    // decouples wall-clock time from render progress (preview pacing breaks).
    // Blocking here keeps at most kMaxInFlight buffers pending while the GPU
    // always has work queued.
    static constexpr size_t kMaxInFlight = 2;

    void submit() override
    {
        if (mCurrent) {
            [mCurrent commit];
            mInFlight.push_back(mCurrent);
            mCurrent = nil;
        }
        while (mInFlight.size() > kMaxInFlight) {
            [mInFlight.front() waitUntilCompleted];
            mInFlight.pop_front();
        }
    }

    void waitIdle() override
    {
        submit();
        if (!mInFlight.empty()) {
            [mInFlight.back() waitUntilCompleted];  // in-order queue: back covers all
            mInFlight.clear();
        }
    }

private:
    id<MTLCommandBuffer> current()
    {
        if (!mCurrent)
            mCurrent = [mQueue commandBuffer];
        return mCurrent;
    }
    id<MTLCommandQueue> mQueue;
    id<MTLCommandBuffer> mCurrent = nil;
    std::deque<id<MTLCommandBuffer>> mInFlight;
};

// Swizzle-free upload of the RGBA8 present target into the drawable: the
// texture write consumes float4 RGBA regardless of the layer's BGRA storage.
static NSString* const kPresentBlitSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void rhi_present_blit(constant uint2& dims [[buffer(0)]],
                             device const uchar4* src [[buffer(1)]],
                             texture2d<float, access::write> dst [[texture(0)]],
                             uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dims.x || gid.y >= dims.y)
        return;
    // The drawable displays row 0 at the top of the window (verified on
    // macOS 15 — no Core Animation flip applies to CAMetalLayer drawables),
    // matching the present-target convention. Straight copy.
    dst.write(float4(src[gid.y * dims.x + gid.x]) / 255.0f, gid);
}
)MSL";

class MetalDevice final : public Device {
public:
    explicit MetalDevice(const DeviceDesc& desc)
    {
        mDev = MTLCreateSystemDefaultDevice();
        if (!mDev)
            throw std::runtime_error("no Metal device available");
        mQueue = [mDev newCommandQueue];
        if (!desc.shaderSource.empty()) {
            NSError* err = nil;
            MTLCompileOptions* opts = [MTLCompileOptions new];
            mLib = [mDev newLibraryWithSource:[NSString stringWithUTF8String:desc.shaderSource.c_str()]
                                      options:opts
                                        error:&err];
            if (!mLib)
                throw std::runtime_error("MSL compilation failed:\n" + nsErrorToString(err));
        }
    }

    Capabilities capabilities() const override
    {
        return Capabilities{ BackendKind::Metal, (bool)[mDev supportsRaytracing], true };
    }

    std::unique_ptr<Buffer> createBuffer(const BufferDesc& desc) override
    {
        return std::make_unique<MetalBuffer>(mDev, desc);
    }
    std::unique_ptr<ComputePipeline> createPipeline(const ComputePipelineDesc& desc) override
    {
        if (!mLib)
            throw std::runtime_error("device created without shader source");
        return std::make_unique<MetalComputePipeline>(mDev, mLib, desc);
    }
    std::unique_ptr<CommandStream> createStream() override
    {
        // All streams and the present blit share one queue: command buffers on
        // a queue start in commit order and default hazard tracking covers the
        // present-target buffer, so present-after-submit needs no fences.
        return std::make_unique<MetalCommandStream>(mQueue);
    }

    std::unique_ptr<Texture> createTexture(const TextureDesc&) override
    {
        throw std::logic_error("Metal textures land in M3 (textured scenes)");
    }
    std::unique_ptr<RayIntersector> createIntersector() override;

    Buffer& presentTarget(int width, int height) override
    {
        if (!mWindow)
            createWindow(width, height);
        else if (width != mPresentW || height != mPresentH)
            throw std::logic_error("present target resize not supported");
        return *mPresentBuf;
    }

    bool present() override
    {
        if (!mWindow)
            throw std::logic_error("present() before presentTarget()");
        @autoreleasepool {
            for (;;) {
                NSEvent* ev = [NSApp nextEventMatchingMask:NSEventMaskAny
                                                 untilDate:[NSDate distantPast]
                                                    inMode:NSDefaultRunLoopMode
                                                   dequeue:YES];
                if (!ev)
                    break;
                if (ev.type == NSEventTypeKeyDown) {
                    NSString* ch = ev.charactersIgnoringModifiers;
                    if (ev.keyCode == 53 /* Esc */ || [ch isEqualToString:@"q"]) {
                        mWinDelegate.closeRequested = YES;
                        continue;
                    }
                }
                [NSApp sendEvent:ev];
            }
            if (mWinDelegate.closeRequested)
                return false;

            id<CAMetalDrawable> drawable = [mLayer nextDrawable];
            if (!drawable)
                return true;  // transient (e.g. window occluded); keep going
            id<MTLCommandBuffer> cb = [mQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:mBlitPso];
            uint32_t dims[2] = { (uint32_t)mPresentW, (uint32_t)mPresentH };
            [enc setBytes:dims length:sizeof(dims) atIndex:0];
            [enc setBuffer:static_cast<MetalBuffer*>(mPresentBuf.get())->handle()
                    offset:0
                   atIndex:1];
            [enc setTexture:drawable.texture atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((mPresentW + 15) / 16, (mPresentH + 15) / 16, 1)
                threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [enc endEncoding];
            [cb presentDrawable:drawable];
            [cb commit];
        }
        return true;
    }

private:
    void createWindow(int width, int height)
    {
        // Cocoa bring-up for a non-bundled CLI binary; must run on the main
        // thread (it does: the renderer drives present() from main).
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        [NSApp finishLaunching];

        mWindow = [[NSWindow alloc]
            initWithContentRect:NSMakeRect(0, 0, width, height)
                      styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                                 | NSWindowStyleMaskMiniaturizable)
                        backing:NSBackingStoreBuffered
                          defer:NO];
        mWindow.releasedWhenClosed = NO;
        mWindow.title = @"FluoraMini";
        mWinDelegate = [RhiWindowDelegate new];
        mWindow.delegate = mWinDelegate;

        mLayer = [CAMetalLayer layer];
        mLayer.device = mDev;
        mLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        mLayer.framebufferOnly = NO;  // written by the blit compute kernel
        mLayer.drawableSize = CGSizeMake(width, height);
        mWindow.contentView.wantsLayer = YES;
        mWindow.contentView.layer = mLayer;

        [mWindow center];
        [mWindow makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];

        NSError* err = nil;
        id<MTLLibrary> lib = [mDev newLibraryWithSource:kPresentBlitSrc
                                                options:[MTLCompileOptions new]
                                                  error:&err];
        id<MTLFunction> fn = [lib newFunctionWithName:@"rhi_present_blit"];
        if (fn)
            mBlitPso = [mDev newComputePipelineStateWithFunction:fn error:&err];
        if (!mBlitPso)
            throw std::runtime_error("present blit pipeline failed: " + nsErrorToString(err));

        mPresentBuf = std::make_unique<MetalBuffer>(
            mDev, BufferDesc{ (size_t)width * height * 4, MemoryLocation::DeviceLocal,
                              "rhi.present" });
        mPresentW = width;
        mPresentH = height;
    }

    id<MTLDevice> mDev;
    id<MTLCommandQueue> mQueue;
    id<MTLLibrary> mLib = nil;

    NSWindow* mWindow = nil;
    CAMetalLayer* mLayer = nil;
    RhiWindowDelegate* mWinDelegate = nil;
    id<MTLComputePipelineState> mBlitPso = nil;
    std::unique_ptr<Buffer> mPresentBuf;
    int mPresentW = 0, mPresentH = 0;
};

// Compute-traversal intersector: uploads the CPU-built threaded BVH; the
// paired device code is rt_closest_hit in raytrace.metal. The M5 hardware-RT
// variant replaces this class without touching kernel call sites.
class MetalRayIntersector final : public RayIntersector {
public:
    explicit MetalRayIntersector(id<MTLDevice> dev) : mDev(dev) {}

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
        // Metal rejects zero-length buffers; empty scenes get a dummy that
        // numNodes()==0 guards against ever being read.
        auto buf = std::make_unique<MetalBuffer>(
            mDev, BufferDesc{ std::max<size_t>(bytes, 16), MemoryLocation::Shared, name });
        if (data && bytes)
            memcpy(buf->hostPtr(), data, bytes);
        return buf;
    }
    id<MTLDevice> mDev;
    uint32_t mNumNodes = 0;
    std::unique_ptr<Buffer> mNodes, mTris, mPositions;
};

std::unique_ptr<RayIntersector> MetalDevice::createIntersector()
{
    return std::make_unique<MetalRayIntersector>(mDev);
}

std::unique_ptr<Device> createDevice(BackendKind kind, const DeviceDesc& desc)
{
    if (kind != BackendKind::Metal)
        throw std::logic_error("only the Metal backend is available on this platform");
    return std::make_unique<MetalDevice>(desc);
}

} // namespace rhi
