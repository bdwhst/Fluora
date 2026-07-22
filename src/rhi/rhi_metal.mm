// Metal backend for the RHI (see docs/metal-rhi-design.md, milestone M1).
// Implements Device / Buffer / ComputePipeline / CommandStream. Texture,
// RayIntersector and presentation are M3/M4 work and throw for now.
// Compiled with ARC (-fobjc-arc); ObjC objects held as C++ members are strong.
#import <Metal/Metal.h>

#include "rhi.h"

#include <stdexcept>
#include <string>

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
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:desc.entryPoint.c_str()]];
        if (!fn)
            throw std::runtime_error("Metal kernel not found: " + desc.entryPoint);
        NSError* err = nil;
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

    void submit() override
    {
        if (mCurrent) {
            [mCurrent commit];
            mLastCommitted = mCurrent;
            mCurrent = nil;
        }
    }

    void waitIdle() override
    {
        submit();
        if (mLastCommitted) {
            [mLastCommitted waitUntilCompleted];
            mLastCommitted = nil;
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
    id<MTLCommandBuffer> mLastCommitted = nil;
};

class MetalDevice final : public Device {
public:
    explicit MetalDevice(const DeviceDesc& desc)
    {
        mDev = MTLCreateSystemDefaultDevice();
        if (!mDev)
            throw std::runtime_error("no Metal device available");
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
        return std::make_unique<MetalCommandStream>([mDev newCommandQueue]);
    }

    std::unique_ptr<Texture> createTexture(const TextureDesc&) override
    {
        throw std::logic_error("Metal textures land in M3 (textured scenes)");
    }
    std::unique_ptr<RayIntersector> createIntersector() override
    {
        throw std::logic_error("RayIntersector lands in M3 (BVH scenes)");
    }
    Buffer& presentTarget(int, int) override
    {
        throw std::logic_error("presentation lands in M4 (interactive preview)");
    }
    void present() override
    {
        throw std::logic_error("presentation lands in M4 (interactive preview)");
    }

private:
    id<MTLDevice> mDev;
    id<MTLLibrary> mLib = nil;
};

std::unique_ptr<Device> createDevice(BackendKind kind, const DeviceDesc& desc)
{
    if (kind != BackendKind::Metal)
        throw std::logic_error("only the Metal backend is available on this platform");
    return std::make_unique<MetalDevice>(desc);
}

} // namespace rhi
