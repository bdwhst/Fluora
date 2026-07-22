// Device-side access to the RHI bindless texture heap (Device::textureHeap).
// The heap buffer holds one 64-bit entry per texture, indexed by
// Texture::shaderHandle(): MTLResourceID on Metal (read directly as a texture
// — Metal 3 bindless, no argument encoder), cudaTextureObject_t on CUDA (the
// M4 counterpart shims tex_heap_sample onto tex2D). Same layout, same kernel
// code on both backends.
//
// Sampler state is fixed here — bilinear + wrap — matching the
// cudaTextureDesc set up by Scene::LoadTextureFromMemory.
// Concatenated after the shared headers; do not #include.

struct RhiTex {
    metal::texture2d<float> t;
};

inline float4 tex_heap_sample(device const RhiTex* heap, uint idx, float2 uv)
{
    constexpr metal::sampler s(metal::address::repeat, metal::filter::linear);
    return heap[idx].t.sample(s, uv);
}
