#pragma once
// Backend-agnostic parallel primitives built on the RHI (design doc M2).
// Host code here dispatches only named kernels through rhi:: interfaces
// (invariant I-4); the kernel implementations live in primitives_gpu.h, and the
// CUDA backend later registers same-named equivalents.
//
// Element type is uint32 for now — enough for flags/indices/keys, which is
// what the renderer needs (path compaction, material keys). Temp buffers are
// allocated per call; cache them when these move into the frame loop.
#include "rhi.h"

#include <cstdint>
#include <memory>

namespace rhi {

class Algorithms {
public:
    explicit Algorithms(Device& device);

    // result (a >=4-byte buffer) must be zeroed by the caller beforehand;
    // the kernel accumulates into it atomically.
    void reduceSum(CommandStream& stream, Buffer& in, uint32_t n, Buffer& result);

    // out[i] = sum of in[0..i). in/out may not alias.
    void exclusiveScan(CommandStream& stream, Buffer& in, uint32_t n, Buffer& out);

    // Packs in[i] where flags[i] != 0 into out, preserving order. Synchronizes
    // the stream (needs the scanned flags on the host for the count) and
    // returns the number of kept elements. flags must be Shared-memory.
    uint32_t compact(CommandStream& stream, Buffer& in, Buffer& flags, uint32_t n, Buffer& out);

    // In-place stable LSD radix sort of n uint32 keys.
    void radixSort(CommandStream& stream, Buffer& keys, uint32_t n);

private:
    Device& mDevice;
    std::unique_ptr<ComputePipeline> mReduce;
    std::unique_ptr<ComputePipeline> mScanBlock;
    std::unique_ptr<ComputePipeline> mScanAdd;
    std::unique_ptr<ComputePipeline> mCompactScatter;
    std::unique_ptr<ComputePipeline> mRadixHist;
    std::unique_ptr<ComputePipeline> mRadixScatter;
};

} // namespace rhi
