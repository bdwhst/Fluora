#include "rhi_algorithms.h"
#include "primitives_shared.h"

#include <stdexcept>

// Lifetime note: temp buffers created here are released (host-side) before the
// GPU may have executed the recorded dispatches. This relies on the backend
// keeping recorded resources alive until command completion — Metal command
// buffers retain referenced MTLBuffers; the CUDA backend must defer frees the
// same way (e.g. free-queue drained on stream sync).

namespace rhi {

namespace {

Dim3 gridFor(uint32_t n)
{
    return Dim3{ (n + PRIM_TILE - 1) / PRIM_TILE, 1, 1 };
}
constexpr Dim3 kBlock{ PRIM_TILE, 1, 1 };

PrimParams makeParams(uint32_t n, uint32_t shift = 0)
{
    PrimParams p = {};
    p.n = n;
    p.numBlocks = gridFor(n).x;
    p.shift = shift;
    return p;
}

} // namespace

Algorithms::Algorithms(Device& device) : mDevice(device)
{
    mReduce = device.createPipeline({ "prim_reduce_sum" });
    mScanBlock = device.createPipeline({ "prim_scan_block" });
    mScanAdd = device.createPipeline({ "prim_scan_add_offsets" });
    mCompactScatter = device.createPipeline({ "prim_compact_scatter" });
    mRadixHist = device.createPipeline({ "prim_radix_histogram" });
    mRadixScatter = device.createPipeline({ "prim_radix_scatter" });
}

void Algorithms::reduceSum(CommandStream& stream, Buffer& in, uint32_t n, Buffer& result)
{
    if (n == 0)
        return;
    PrimParams p = makeParams(n);
    stream.dispatch(*mReduce, gridFor(n), kBlock, &p, sizeof(p), { &in, &result });
}

void Algorithms::exclusiveScan(CommandStream& stream, Buffer& in, uint32_t n, Buffer& out)
{
    if (n == 0)
        return;
    PrimParams p = makeParams(n);
    uint32_t numBlocks = p.numBlocks;
    auto blockSums = mDevice.createBuffer(
        { (size_t)numBlocks * sizeof(uint32_t), MemoryLocation::DeviceLocal, "scan.blockSums" });
    stream.dispatch(*mScanBlock, gridFor(n), kBlock, &p, sizeof(p), { &in, &out, blockSums.get() });
    if (numBlocks > 1) {
        auto scannedSums = mDevice.createBuffer(
            { (size_t)numBlocks * sizeof(uint32_t), MemoryLocation::DeviceLocal, "scan.scannedSums" });
        exclusiveScan(stream, *blockSums, numBlocks, *scannedSums);
        stream.dispatch(*mScanAdd, gridFor(n), kBlock, &p, sizeof(p), { &out, scannedSums.get() });
    }
    // Temps die here with dispatches still recorded — covered by the lifetime
    // note above, same as compact()/radixSort().
}

uint32_t Algorithms::compact(CommandStream& stream, Buffer& in, Buffer& flags, uint32_t n, Buffer& out)
{
    if (n == 0)
        return 0;
    auto scannedFlags = mDevice.createBuffer(
        { (size_t)n * sizeof(uint32_t), MemoryLocation::Shared, "compact.scannedFlags" });
    exclusiveScan(stream, flags, n, *scannedFlags);
    PrimParams p = makeParams(n);
    stream.dispatch(*mCompactScatter, gridFor(n), kBlock, &p, sizeof(p),
                    { &in, &flags, scannedFlags.get(), &out });
    stream.waitIdle();
    const uint32_t* flagsHost = (const uint32_t*)flags.hostPtr();
    const uint32_t* scannedHost = (const uint32_t*)scannedFlags->hostPtr();
    if (!flagsHost || !scannedHost)
        throw std::runtime_error("compact: flags buffer must be host-visible (Shared)");
    return scannedHost[n - 1] + (flagsHost[n - 1] != 0 ? 1u : 0u);
}

void Algorithms::radixSort(CommandStream& stream, Buffer& keys, uint32_t n)
{
    if (n <= 1)
        return;
    uint32_t numBlocks = gridFor(n).x;
    uint32_t histSize = PRIM_RADIX_DIGITS * numBlocks;
    auto tmp = mDevice.createBuffer(
        { (size_t)n * sizeof(uint32_t), MemoryLocation::DeviceLocal, "sort.tmp" });
    auto hist = mDevice.createBuffer(
        { (size_t)histSize * sizeof(uint32_t), MemoryLocation::DeviceLocal, "sort.hist" });
    auto histScanned = mDevice.createBuffer(
        { (size_t)histSize * sizeof(uint32_t), MemoryLocation::DeviceLocal, "sort.histScanned" });

    Buffer* src = &keys;
    Buffer* dst = tmp.get();
    static_assert(32 % PRIM_RADIX_BITS == 0);
    for (uint32_t shift = 0; shift < 32; shift += PRIM_RADIX_BITS) {
        PrimParams p = makeParams(n, shift);
        stream.dispatch(*mRadixHist, gridFor(n), kBlock, &p, sizeof(p), { src, hist.get() });
        exclusiveScan(stream, *hist, histSize, *histScanned);
        stream.dispatch(*mRadixScatter, gridFor(n), kBlock, &p, sizeof(p),
                        { src, dst, histScanned.get() });
        std::swap(src, dst);
    }
    // 32/PRIM_RADIX_BITS = 8 passes: even number of swaps, result is in `keys`.
}

} // namespace rhi
