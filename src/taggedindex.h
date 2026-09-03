#pragma once
#include <cstdint>
#include <cassert>
#include <vector>
#include <tuple>
#include <cuda_runtime.h>
#include "taggedptr.h"
#include "memoryUtils.h"

// Phase-2 portability machinery: persistent references to host-built polymorphic
// objects are stored as {type tag, array index} handles instead of raw pointers.
// Raw device pointers only exist transiently *inside* a kernel, resolved from a
// TypedPoolView that the backend binds per dispatch. This is the invariant a
// non-unified-memory backend (Metal: MTLBuffer base + offset, gpuAddress != host
// address) needs; CUDA resolves against the same view at zero extra cost.
//
// Tag numbering is identical to TaggedPointer<Ts...>: 0 = null, 1 + IndexOf<T>.

template <typename... Ts>
class TaggedIndex
{
public:
    using Types = TypePack<Ts...>;

    template <typename T>
    __device__ __host__ static constexpr unsigned int TypeIndex()
    {
        using Tp = typename std::remove_cv_t<T>;
        if constexpr (std::is_same_v<Tp, std::nullptr_t>)
            return 0;
        else
            return 1 + IndexOf<Tp, Types>::count;
    }

    TaggedIndex() = default;
    __device__ __host__ TaggedIndex(std::nullptr_t) {}

    template <typename T>
    __device__ __host__ static TaggedIndex make(uint32_t index)
    {
        TaggedIndex ti;
        ti.bits = ((uint32_t)TypeIndex<T>() << tagShift) | (index & indexMask);
        return ti;
    }

    __device__ __host__ bool operator==(const TaggedIndex& t) const { return bits == t.bits; }
    __device__ __host__ bool operator!=(const TaggedIndex& t) const { return bits != t.bits; }
    __device__ __host__ explicit operator bool() const { return bits != 0; }

    __device__ __host__ unsigned int Tag() const { return bits >> tagShift; }
    __device__ __host__ uint32_t Index() const { return bits & indexMask; }
    template <typename T>
    __device__ __host__ bool Is() const { return Tag() == TypeIndex<T>(); }
    static constexpr unsigned int MaxTag() { return sizeof...(Ts); }

private:
    static constexpr int tagShift = 27;  // 5 tag bits / 27 index bits
    static constexpr uint32_t indexMask = (1u << tagShift) - 1;
    uint32_t bits = 0;
};

// POD view over type-segregated device arrays; embeddable in kernel parameter
// structs (e.g. SceneInfoDev). resolve() yields a transient tagged pointer (any
// TaggedPointer<Ts...>-derived PtrT), valid only within the current dispatch.
template <typename... Ts>
struct TypedPoolView
{
    void* bases[sizeof...(Ts)] = {};

    // Element stride of pool slot i (sizeof of the i-th type). A pack fold
    // rather than a static constexpr array: indexing such an array ODR-uses it,
    // which nvcc rejects in device code ("identifier undefined in device code").
    __device__ __host__ static constexpr size_t typeSize(unsigned int i)
    {
        size_t s = 0;
        unsigned int k = 0;
        ((k++ == i ? (s = sizeof(Ts), 0) : 0), ...);
        return s;
    }

    template <typename PtrT>
    __device__ __host__ PtrT resolve(TaggedIndex<Ts...> handle) const
    {
        unsigned int tag = handle.Tag();
        if (tag == 0)
            return PtrT(nullptr);
        void* ptr = (char*)bases[tag - 1] + (size_t)handle.Index() * typeSize(tag - 1);
        return PtrT(ptr, tag);
    }
};

// Host-side pool: collects concrete objects by value during scene load, then
// uploads them as flat per-type arrays. Objects are copied bitwise: pointers
// *inside* them (SpectrumPtr, Distribution2D*, ...) still rely on unified
// memory and are the next conversion targets.
template <typename... Ts>
class TypedPoolBuilder
{
public:
    using Handle = TaggedIndex<Ts...>;
    using View = TypedPoolView<Ts...>;

    template <typename T>
    Handle add(const T& obj)
    {
        auto& vec = std::get<std::vector<T>>(mData);
        vec.push_back(obj);
        return Handle::template make<T>((uint32_t)vec.size() - 1);
    }

    template <typename T>
    T* host_get(Handle h)
    {
        assert(h.template Is<T>());
        return &std::get<std::vector<T>>(mData)[h.Index()];
    }
    template <typename T>
    const T* host_get(Handle h) const
    {
        assert(h.template Is<T>());
        return &std::get<std::vector<T>>(mData)[h.Index()];
    }

    View upload(Allocator alloc)
    {
        View view = {};
        uploadAll<0>(view, alloc);
        return view;
    }

private:
    template <size_t I>
    void uploadAll(View& view, Allocator alloc)
    {
        if constexpr (I < sizeof...(Ts))
        {
            using T = std::tuple_element_t<I, std::tuple<Ts...>>;
            auto& vec = std::get<I>(mData);
            if (!vec.empty())
            {
                T* dev = alloc.allocate<T>(vec.size());
                cudaMemcpy(dev, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice);
                view.bases[I] = dev;
            }
            uploadAll<I + 1>(view, alloc);
        }
    }

    std::tuple<std::vector<Ts>...> mData;
};
