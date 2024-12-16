#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <utility>
#include <unordered_map>
#include <string>
#include <iostream>

class MemoryResourceBackend
{
public:
    virtual ~MemoryResourceBackend() = default;
    // Assume bytes != 0
    virtual void* allocate_bytes(size_t bytes, size_t alignment) = 0;
    virtual void free_bytes(void* ptr, size_t bytes, size_t alignment) = 0;
    std::unordered_map<void*, size_t> ptrs;
};

class MainMemoryResourceBackend : public MemoryResourceBackend
{
private:
    MainMemoryResourceBackend() = default;
    MainMemoryResourceBackend(const MainMemoryResourceBackend&) = delete;
    MainMemoryResourceBackend& operator=(const MainMemoryResourceBackend&) = delete;
public:
    virtual void* allocate_bytes(size_t bytes, size_t alignment) override
    {
        void* ptr = nullptr;
        ptr = _aligned_malloc(bytes, alignment);
        if (ptr) ptrs[ptr] = bytes;
        return ptr;
    }
    virtual void free_bytes(void* ptr, size_t bytes, size_t alignment) override
    {
        if (!ptr) return;
        if (ptrs.count(ptr))
        {
            _aligned_free(ptr);
            ptrs.erase(ptr);
        }
        else
        {
            std::cout << "ERROR: FREEING A INVALID MEM ADDR " << ptr << std::endl;
            throw std::runtime_error("Invalid free");
        }
    }
    static MainMemoryResourceBackend* getInstance()
    {
        static MainMemoryResourceBackend* instance = nullptr;
        if (instance == nullptr)
        {
            instance = new MainMemoryResourceBackend();
        }
        return instance;
    }
};

class CUDAMemoryResourceBackend : public MemoryResourceBackend
{
private:
    CUDAMemoryResourceBackend() = default;
    CUDAMemoryResourceBackend(const CUDAMemoryResourceBackend&) = delete;
    CUDAMemoryResourceBackend& operator=(const CUDAMemoryResourceBackend&) = delete;
public:
    virtual void* allocate_bytes(size_t bytes, size_t alignment) override
    {
        void* ptr = nullptr;
        cudaError_t err = cudaMallocManaged(&ptr, bytes, cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMallocManaged failed: " + std::string(cudaGetErrorString(err)));
        }
        if (ptr) ptrs[ptr] = bytes;
        return ptr;
    }
    virtual void free_bytes(void* ptr, size_t bytes, size_t alignment) override
    {
        if (!ptr) return;
        if (ptrs.count(ptr))
        {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaFree failed: " + std::string(cudaGetErrorString(err)));
            }
            ptrs.erase(ptr);
        }
        else
        {
            std::cout << "ERROR: FREEING A INVALID MEM ADDR " << ptr << std::endl;
            throw std::runtime_error("Invalid free");
        }
    }
    static CUDAMemoryResourceBackend* getInstance()
    {
        static CUDAMemoryResourceBackend* instance = nullptr;
        if (instance == nullptr)
        {
            instance = new CUDAMemoryResourceBackend();
        }
        return instance;
    }
};


class MonotonicBlockMemoryResourceBackend : public MemoryResourceBackend
{
public:
    explicit MonotonicBlockMemoryResourceBackend(size_t blocksize, MemoryResourceBackend* backend): mBlockSize(blocksize), mCurrent(nullptr), mBlockList(nullptr), mCurrentPos(0), mUpstream(backend) {}
    virtual void* allocate_bytes(size_t bytes, size_t align) override
    {
        //std::cout << "allocate_bytes: bytes " << bytes << " align " << align << std::endl;
        cudaDeviceSynchronize();
        if (bytes > mBlockSize) return mUpstream->allocate_bytes(bytes, align);
        if ((mCurrentPos % align) != 0)
            mCurrentPos += align - (mCurrentPos % align);
        if (!mCurrent || mCurrentPos + bytes > mCurrent->size)
        {
            mCurrent = allocate_block(mBlockSize);
            mCurrentPos = 0;
        }
        void* ptr = (char*)mCurrent->ptr + mCurrentPos;
        mCurrentPos += bytes;
        return ptr;
    }
    virtual void free_bytes(void* ptr, size_t bytes, size_t alignment) override
    {
        if (!ptr) return;
        if (bytes > mBlockSize) mUpstream->free_bytes(ptr, bytes, alignment);
    }
    ~MonotonicBlockMemoryResourceBackend() {
        block* b = mBlockList;
        while (b)
        {
            block* next = b->next;
            free_block(b);
            b = next;
        }
        mCurrent = nullptr;
        mBlockList = nullptr;
        mCurrentPos = 0;
    }
private:
    struct block {
        void* ptr;
        size_t size;
        block* next;
    };

    void free_block(block* b)
    {
        mUpstream->free_bytes(b, sizeof(block) + b->size, alignof(block));
    }

    block* allocate_block(size_t size)
    {
        block* b = (block*)(mUpstream->allocate_bytes(sizeof(block) + size, alignof(block)));
        b->ptr = ((char*)b) + sizeof(block);
        b->size = size;
        b->next = mBlockList;
        mBlockList = b;
        return b;
    }

    size_t mBlockSize;
    block* mCurrent;
    block* mBlockList;
    size_t mCurrentPos;
    MemoryResourceBackend* mUpstream;

};

class Allocator
{
public:
    Allocator():mBackend(MainMemoryResourceBackend::getInstance()){}
    explicit Allocator(MemoryResourceBackend* backend) :mBackend(backend) {}
    ~Allocator() {  }
    template<typename T>
    T* allocate(size_t count)
    {
        size_t size = count * sizeof(T);
        if (size == 0) return nullptr;
        void* ptr;
        ptr = mBackend->allocate_bytes(size, alignof(T));
        return (T*)ptr;
    }
    void* allocate_bytes(size_t size, size_t alignment)
    {
        if (size == 0) return nullptr;
        return mBackend->allocate_bytes(size, alignment);
    }
    void deallocate_bytes(void* ptr, size_t size, size_t alignment)
    {
        mBackend->free_bytes(ptr, size, alignment);
    }
    template<typename T>
    void deallocate(T* p, size_t count)
    {
        mBackend->free_bytes(p, sizeof(T) * count, alignof(T));
    }
    template <class T, class... Args>
    T* new_object(Args &&...args) {
        // NOTE: this doesn't handle constructors that throw exceptions...
        T* p = allocate<T>(1);
        construct(p, std::forward<Args>(args)...);
        return p;
    }
    template <class T>
    void delete_object(T* p) {
        destroy(p);
        deallocate(p);
    }
    template <class T, class... Args>
    void construct(T* p, Args &&...args) {
        ::new ((void*)p) T(std::forward<Args>(args)...);
    }
    template <class T>
    void destroy(T* p) {
        p->~T();
    }
private:
    MemoryResourceBackend* mBackend;
};