#ifndef RHI_TEXTURE_GPU_H
#define RHI_TEXTURE_GPU_H
// Device-side access to the RHI bindless texture heap (Device::textureHeap).
// The heap buffer holds one 64-bit entry per texture, indexed by
// Texture::shaderHandle(): MTLResourceID on Metal (read directly as a texture
// — Metal 3 bindless, no argument encoder), cudaTextureObject_t on CUDA
// (tex2D through the object). Same layout, same kernel code on both backends;
// this is the one shared-code file whose function BODY is per backend
// (docs/portable-device-code.md §2: genuinely different APIs behind one name).
//
// Sampler state is fixed here — bilinear + wrap, normalized coordinates —
// matching the cudaTextureDesc set up by Scene::LoadTextureFromMemory.
// Concatenated after gpu_portable.h under MSL; #include-able elsewhere.

#if defined(__METAL_VERSION__)

struct RhiTex {
    metal::texture2d<float> t;
};

inline float4 tex_heap_sample(device const RhiTex* heap, uint idx, float2 uv)
{
    constexpr metal::sampler s(metal::address::repeat, metal::filter::linear);
    return heap[idx].t.sample(s, uv);
}

#elif defined(__CUDACC__)

#include "gpu_portable.h"

struct RhiTex {
    cudaTextureObject_t t;
};

GPU_FN inline gpu_float4 tex_heap_sample(const RhiTex* heap, uint idx, gpu_float2 uv)
{
    float4 v = tex2D<float4>(heap[idx].t, uv.x, uv.y);
    return gpu_float4(v.x, v.y, v.z, v.w);
}

#else

// Host personality: the heap entry keeps its 8-byte layout for struct-size
// parity, but there is no sampler — host-compiled shared code must not reach
// a texture read (SharedHostTest doesn't; a CPU-debug path would need a
// software bilinear sampler here).
#include "gpu_portable.h"

struct RhiTex {
    unsigned long long t;
};

GPU_FN inline gpu_float4 tex_heap_sample(const RhiTex*, uint, gpu_float2)
{
    return gpu_float4(0.0f, 0.0f, 0.0f, 1.0f);
}

#endif

#endif // RHI_TEXTURE_GPU_H
