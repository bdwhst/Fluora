#pragma once
#include <type_traits>
#include <cuda_runtime.h>
namespace detail 
{

    // TaggedPointer Helper Templates
    template <typename F, typename R, typename T>
    __device__ __host__ R Dispatch(F &&func, const void *ptr, int index) 
    {
        return func((const T *)ptr);
    }

    template <typename F, typename R, typename T>
    __device__ __host__ R Dispatch(F &&func, void *ptr, int index) 
    {
        return func((T *)ptr);
    }

    template <typename F, typename R, typename T0, typename T1>
    __device__ __host__ R Dispatch(F &&func, const void *ptr, int index) 
    {
        if (index == 0)
            return func((const T0 *)ptr);
        else
            return func((const T1 *)ptr);
    }

    template <typename F, typename R, typename T0, typename T1>
    __device__ __host__ R Dispatch(F &&func, void *ptr, int index) 
    {
        if (index == 0)
            return func((T0 *)ptr);
        else
            return func((T1 *)ptr);
    }

    template <typename F, typename R, typename T0, typename T1, typename T2>
    __device__ __host__ R Dispatch(F &&func, const void *ptr, int index) 
    {
        switch (index) {
        case 0:
            return func((const T0 *)ptr);
        case 1:
            return func((const T1 *)ptr);
        default:
            return func((const T2 *)ptr);
        }
    }

    template <typename F, typename R, typename T0, typename T1, typename T2>
    __device__ __host__ R Dispatch(F &&func, void *ptr, int index) 
    {
        switch (index) {
        case 0:
            return func((T0 *)ptr);
        case 1:
            return func((T1 *)ptr);
        default:
            return func((T2 *)ptr);
        }
    }

    template <typename F, typename R, typename T0, typename T1, typename T2, typename T3>
    __device__ __host__ R Dispatch(F &&func, const void *ptr, int index) 
    {
        switch (index) {
        case 0:
            return func((const T0 *)ptr);
        case 1:
            return func((const T1 *)ptr);
        case 2:
            return func((const T2 *)ptr);
        default:
            return func((const T3 *)ptr);
        }
    }

    template <typename F, typename R, typename T0, typename T1, typename T2, typename T3>
    __device__ __host__ R Dispatch(F &&func, void *ptr, int index) 
    {
        switch (index) {
        case 0:
            return func((T0 *)ptr);
        case 1:
            return func((T1 *)ptr);
        case 2:
            return func((T2 *)ptr);
        default:
            return func((T3 *)ptr);
        }
    }

    template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
            typename T4>
    __device__ __host__ R Dispatch(F &&func, const void *ptr, int index) 
    {
        switch (index) {
        case 0:
            return func((const T0 *)ptr);
        case 1:
            return func((const T1 *)ptr);
        case 2:
            return func((const T2 *)ptr);
        case 3:
            return func((const T3 *)ptr);
        default:
            return func((const T4 *)ptr);
        }
    }

    template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
            typename T4>
    __device__ __host__ R Dispatch(F &&func, void *ptr, int index) 
    {
        switch (index) {
        case 0:
            return func((T0 *)ptr);
        case 1:
            return func((T1 *)ptr);
        case 2:
            return func((T2 *)ptr);
        case 3:
            return func((T3 *)ptr);
        default:
            return func((T4 *)ptr);
        }
    }

    template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
            typename T4, typename T5>
    __device__ __host__ R Dispatch(F &&func, const void *ptr, int index) 
    {
        switch (index) {
        case 0:
            return func((const T0 *)ptr);
        case 1:
            return func((const T1 *)ptr);
        case 2:
            return func((const T2 *)ptr);
        case 3:
            return func((const T3 *)ptr);
        case 4:
            return func((const T4 *)ptr);
        default:
            return func((const T5 *)ptr);
        }
    }

    template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
            typename T4, typename T5>
    __device__ __host__ R Dispatch(F &&func, void *ptr, int index) 
    {
        switch (index) {
        case 0:
            return func((T0 *)ptr);
        case 1:
            return func((T1 *)ptr);
        case 2:
            return func((T2 *)ptr);
        case 3:
            return func((T3 *)ptr);
        case 4:
            return func((T4 *)ptr);
        default:
            return func((T5 *)ptr);
        }
    }

    template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
            typename T4, typename T5, typename T6>
    __device__ __host__ R Dispatch(F &&func, const void *ptr, int index) 
    {
        switch (index) {
        case 0:
            return func((const T0 *)ptr);
        case 1:
            return func((const T1 *)ptr);
        case 2:
            return func((const T2 *)ptr);
        case 3:
            return func((const T3 *)ptr);
        case 4:
            return func((const T4 *)ptr);
        case 5:
            return func((const T5 *)ptr);
        default:
            return func((const T6 *)ptr);
        }
    }

    template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
            typename T4, typename T5, typename T6>
    __device__ __host__ R Dispatch(F &&func, void *ptr, int index) 
    {
        switch (index) {
        case 0:
            return func((T0 *)ptr);
        case 1:
            return func((T1 *)ptr);
        case 2:
            return func((T2 *)ptr);
        case 3:
            return func((T3 *)ptr);
        case 4:
            return func((T4 *)ptr);
        case 5:
            return func((T5 *)ptr);
        default:
            return func((T6 *)ptr);
        }
    }

    template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
            typename T4, typename T5, typename T6, typename T7>
    __device__ __host__ R Dispatch(F &&func, const void *ptr, int index) 
    {
        switch (index) {
        case 0:
            return func((const T0 *)ptr);
        case 1:
            return func((const T1 *)ptr);
        case 2:
            return func((const T2 *)ptr);
        case 3:
            return func((const T3 *)ptr);
        case 4:
            return func((const T4 *)ptr);
        case 5:
            return func((const T5 *)ptr);
        case 6:
            return func((const T6 *)ptr);
        default:
            return func((const T7 *)ptr);
        }
    }

    template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
            typename T4, typename T5, typename T6, typename T7>
    __device__ __host__ R Dispatch(F &&func, void *ptr, int index) 
    {
        switch (index) {
        case 0:
            return func((T0 *)ptr);
        case 1:
            return func((T1 *)ptr);
        case 2:
            return func((T2 *)ptr);
        case 3:
            return func((T3 *)ptr);
        case 4:
            return func((T4 *)ptr);
        case 5:
            return func((T5 *)ptr);
        case 6:
            return func((T6 *)ptr);
        default:
            return func((T7 *)ptr);
        }
    }

    template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
            typename T4, typename T5, typename T6, typename T7, typename... Ts,
            typename = typename std::enable_if_t<(sizeof...(Ts) > 0)>>
    __device__ __host__ R Dispatch(F &&func, const void *ptr, int index) 
    {
        switch (index) {
        case 0:
            return func((const T0 *)ptr);
        case 1:
            return func((const T1 *)ptr);
        case 2:
            return func((const T2 *)ptr);
        case 3:
            return func((const T3 *)ptr);
        case 4:
            return func((const T4 *)ptr);
        case 5:
            return func((const T5 *)ptr);
        case 6:
            return func((const T6 *)ptr);
        case 7:
            return func((const T7 *)ptr);
        default:
            return Dispatch<F, R, Ts...>(func, ptr, index - 8);
        }
    }

    template <typename F, typename R, typename T0, typename T1, typename T2, typename T3,
            typename T4, typename T5, typename T6, typename T7, typename... Ts,
            typename = typename std::enable_if_t<(sizeof...(Ts) > 0)>>
    __device__ __host__ R Dispatch(F &&func, void *ptr, int index) 
    {
        switch (index) {
        case 0:
            return func((T0 *)ptr);
        case 1:
            return func((T1 *)ptr);
        case 2:
            return func((T2 *)ptr);
        case 3:
            return func((T3 *)ptr);
        case 4:
            return func((T4 *)ptr);
        case 5:
            return func((T5 *)ptr);
        case 6:
            return func((T6 *)ptr);
        case 7:
            return func((T7 *)ptr);
        default:
            return Dispatch<F, R, Ts...>(func, ptr, index - 8);
        }
    }

    template <typename... Ts>
    struct IsSameType;

    template <typename T>
    struct IsSameType<T> 
    {
        static constexpr bool value = true;
    };

    template <typename T, typename U, typename... Ts>
    struct IsSameType<T, U, Ts...> 
    {
        static constexpr bool value = (std::is_same_v<T, U> && IsSameType<U, Ts...>::value);
    };

    template <typename... Ts>
    struct SameType;
    template <typename T, typename... Ts>
    struct SameType<T, Ts...> 
    {
        using type = T;
        static_assert(IsSameType<T, Ts...>::value, "Not all types in pack are the same");
    };

    template <typename F, typename... Ts>
    struct ReturnType 
    {
        using type = typename SameType<typename std::invoke_result_t<F, Ts *>...>::type;
    };

    template <typename F, typename... Ts>
    struct ReturnTypeConst {
        using type = typename SameType<typename std::invoke_result_t<F, const Ts*>...>::type;
    };
};

template <typename... Ts>
struct TypePack 
{
    static constexpr size_t count = sizeof...(Ts);
};

template <typename T, typename... Ts>
struct IndexOf 
{
    static constexpr int count = 0;
    static_assert(!std::is_same_v<T, T>, "Type not present in TypePack");
};

template <typename T, typename... Ts>
struct IndexOf<T, TypePack<T, Ts...>> 
{
    static constexpr int count = 0;
};

template <typename T, typename U, typename... Ts>
struct IndexOf<T, TypePack<U, Ts...>> 
{
    static constexpr int count = 1 + IndexOf<T, TypePack<Ts...>>::count;
};

template <typename... Ts>
class TaggedPointer 
{
public:
    using Types = TypePack<Ts...>;

    template <typename T>
    static constexpr unsigned int TypeIndex() 
    {
        using Tp = typename std::remove_cv_t<T>;
        if constexpr (std::is_same_v<Tp, std::nullptr_t>) 
            return 0;
        else 
            return 1 + IndexOf<Tp, Types>::count;
    }

    template <typename T>
    __device__ __host__ TaggedPointer(T *ptr)
    {
        uintptr_t iptr = reinterpret_cast<uintptr_t>(ptr);
        constexpr unsigned int type = TypeIndex<T>();
        bits = iptr | ((uintptr_t)type << tagShift);
    }

    __device__ __host__ TaggedPointer(std::nullptr_t np) {}

    __device__ __host__ TaggedPointer(const TaggedPointer &t) { bits = t.bits; }

    __device__ __host__ TaggedPointer &operator=(const TaggedPointer &t)
    {
        bits = t.bits;
        return *this;
    }

    __device__ __host__ TaggedPointer(void* ptr, uint32_t tag)
    {
        uintptr_t iptr = reinterpret_cast<uintptr_t>(ptr);
        bits = iptr | ((uintptr_t)tag << tagShift);
    }

    __device__ __host__ bool operator==(const TaggedPointer &tp) const { return bits == tp.bits; }
    __device__ __host__ bool operator!=(const TaggedPointer &tp) const { return bits != tp.bits; }
    __device__ __host__ explicit operator bool() const { return (bits & ptrMask) != 0; }

    __device__ __host__ unsigned int Tag() const { return ((bits & tagMask) >> tagShift); }
    template <typename T>
    __device__ __host__ bool Is() const { return Tag() == TypeIndex<T>(); }
    static constexpr unsigned int MaxTag() { return sizeof...(Ts); }
    __device__ __host__
    static constexpr unsigned int NumTags() { return MaxTag() + 1; }
    

    template <typename T>
    __device__ __host__ T *Cast()
    {
        return reinterpret_cast<T *>(ptr());
    }
    template <typename T>
    __device__ __host__ const T *Cast() const
    {
        return reinterpret_cast<const T *>(ptr());
    }
    template <typename T>
    __device__ __host__ T *CastOrNullptr()
    {
        if (Is<T>()) 
            return reinterpret_cast<T *>(ptr());
        else 
            return nullptr;
    }
    template <typename T>
    __device__ __host__ const T *CastOrNullptr() const
    {
        if (Is<T>())
            return reinterpret_cast<const T *>(ptr());
        else
            return nullptr;
    }

    __device__ __host__ void *ptr() { return reinterpret_cast<void *>(bits & ptrMask); }
    __device__ __host__ const void *ptr() const { return reinterpret_cast<const void *>(bits & ptrMask); }
    template <typename F>
    __device__ __host__ decltype(auto) Dispatch(F &&func)
    {
        using R = typename detail::ReturnType<F, Ts...>::type;
        return detail::Dispatch<F, R, Ts...>(func, ptr(), Tag() - 1);
    }

    template <typename F>
    __device__ __host__ decltype(auto) Dispatch(F&& func) const
    {
        using R = typename detail::ReturnType<F, Ts...>::type;
        return detail::Dispatch<F, R, Ts...>(func, ptr(), Tag() - 1);
    }
private:
    static constexpr int tagShift = 57;
    static constexpr int tagBits = 64 - tagShift;
    static constexpr uint64_t tagMask = ((1ull << tagBits) - 1) << tagShift;
    static constexpr uint64_t ptrMask = ~tagMask;
    uintptr_t bits = 0;
};