// Parallel primitives for the Metal backend (design doc M2): reduce, exclusive
// scan, compact scatter, LSD radix sort passes, and the simdgroup-aggregated
// work-queue push. Correctness-first implementations; the CUDA backend will
// register same-named kernels (or CUB calls) behind the same host wrappers.
//
// primitives_shared.h is textually prepended before compilation.
// Assumes 32-wide simdgroups and PRIM_TILE (256) threads per threadgroup, one
// element per thread.
using namespace metal;

// Exclusive prefix sum of v across the threadgroup, in thread-index order.
// sgScratch must hold PRIM_TILE/32 entries of threadgroup memory. Contains
// barriers: call from uniform control flow only, and barrier before reusing
// sgScratch.
inline uint tg_exclusive_scan(uint v, uint lid, uint sgId, uint numSg,
                              threadgroup uint* sgScratch)
{
    uint lanePrefix = simd_prefix_exclusive_sum(v);
    uint sgTotal = simd_sum(v);
    if (simd_is_first())
        sgScratch[sgId] = sgTotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgId == 0) {
        uint x = lid < numSg ? sgScratch[lid] : 0u;
        uint p = simd_prefix_exclusive_sum(x);
        if (lid < numSg)
            sgScratch[lid] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return lanePrefix + sgScratch[sgId];
}

// --------------------------------------------------------------------------
// Reduce
// --------------------------------------------------------------------------

kernel void prim_reduce_sum(constant PrimParams& P    [[buffer(0)]],
                            device const uint* in     [[buffer(1)]],
                            device atomic_uint* out   [[buffer(2)]],
                            uint gid  [[thread_position_in_grid]],
                            uint lid  [[thread_index_in_threadgroup]],
                            uint sgId [[simdgroup_index_in_threadgroup]],
                            uint numSg [[simdgroups_per_threadgroup]])
{
    threadgroup uint sgSums[PRIM_TILE / 32u];
    uint v = gid < P.n ? in[gid] : 0u;
    uint s = simd_sum(v);
    if (simd_is_first())
        sgSums[sgId] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        uint total = 0;
        for (uint i = 0; i < numSg; i++)
            total += sgSums[i];
        atomic_fetch_add_explicit(out, total, memory_order_relaxed);
    }
}

// --------------------------------------------------------------------------
// Exclusive scan: per-block scan + block sums, then host recursion over the
// block sums, then offset add. (Multi-dispatch reduce-then-scan; single-pass
// decoupled-lookback needs forward-progress guarantees Apple GPUs don't make.)
// --------------------------------------------------------------------------

kernel void prim_scan_block(constant PrimParams& P    [[buffer(0)]],
                            device const uint* in     [[buffer(1)]],
                            device uint* out          [[buffer(2)]],
                            device uint* blockSums    [[buffer(3)]],
                            uint gid  [[thread_position_in_grid]],
                            uint lid  [[thread_index_in_threadgroup]],
                            uint sgId [[simdgroup_index_in_threadgroup]],
                            uint numSg [[simdgroups_per_threadgroup]],
                            uint groupId [[threadgroup_position_in_grid]])
{
    threadgroup uint sgScratch[PRIM_TILE / 32u];
    uint v = gid < P.n ? in[gid] : 0u;
    uint p = tg_exclusive_scan(v, lid, sgId, numSg, sgScratch);
    if (gid < P.n)
        out[gid] = p;
    if (lid == PRIM_TILE - 1u)
        blockSums[groupId] = p + v;
}

kernel void prim_scan_add_offsets(constant PrimParams& P              [[buffer(0)]],
                                  device uint* out                    [[buffer(1)]],
                                  device const uint* scannedBlockSums [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]],
                                  uint groupId [[threadgroup_position_in_grid]])
{
    if (gid < P.n)
        out[gid] += scannedBlockSums[groupId];
}

// --------------------------------------------------------------------------
// Compact: out[scannedFlags[i]] = in[i] where flags[i] != 0. The caller scans
// the flags first; count = scannedFlags[n-1] + flags[n-1].
// --------------------------------------------------------------------------

kernel void prim_compact_scatter(constant PrimParams& P          [[buffer(0)]],
                                 device const uint* in           [[buffer(1)]],
                                 device const uint* flags        [[buffer(2)]],
                                 device const uint* scannedFlags [[buffer(3)]],
                                 device uint* out                [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]])
{
    if (gid < P.n && flags[gid] != 0u)
        out[scannedFlags[gid]] = in[gid];
}

// --------------------------------------------------------------------------
// LSD radix sort, PRIM_RADIX_BITS per pass. Digit-major histogram layout
// (hist[digit * numBlocks + block]) so one exclusive scan of the whole
// histogram yields stable global scatter bases.
// --------------------------------------------------------------------------

kernel void prim_radix_histogram(constant PrimParams& P  [[buffer(0)]],
                                 device const uint* keys [[buffer(1)]],
                                 device uint* hist       [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]],
                                 uint lid [[thread_index_in_threadgroup]],
                                 uint groupId [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint local[PRIM_RADIX_DIGITS];
    if (lid < PRIM_RADIX_DIGITS)
        atomic_store_explicit(&local[lid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid < P.n) {
        uint d = (keys[gid] >> P.shift) & (PRIM_RADIX_DIGITS - 1u);
        atomic_fetch_add_explicit(&local[d], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < PRIM_RADIX_DIGITS)
        hist[lid * P.numBlocks + groupId] = atomic_load_explicit(&local[lid], memory_order_relaxed);
}

kernel void prim_radix_scatter(constant PrimParams& P          [[buffer(0)]],
                               device const uint* keysIn       [[buffer(1)]],
                               device uint* keysOut            [[buffer(2)]],
                               device const uint* histScanned  [[buffer(3)]],
                               uint gid  [[thread_position_in_grid]],
                               uint lid  [[thread_index_in_threadgroup]],
                               uint sgId [[simdgroup_index_in_threadgroup]],
                               uint numSg [[simdgroups_per_threadgroup]],
                               uint groupId [[threadgroup_position_in_grid]])
{
    threadgroup uint sgScratch[PRIM_TILE / 32u];
    uint key = gid < P.n ? keysIn[gid] : 0u;
    uint d = (key >> P.shift) & (PRIM_RADIX_DIGITS - 1u);
    // Stable rank among this block's earlier same-digit elements: one
    // threadgroup scan per digit value (correctness-first; a ballot-based
    // ranking can replace this if sort ever becomes hot).
    uint rank = 0u;
    for (uint b = 0; b < PRIM_RADIX_DIGITS; b++) {
        uint flag = (gid < P.n && d == b) ? 1u : 0u;
        uint p = tg_exclusive_scan(flag, lid, sgId, numSg, sgScratch);
        if (flag != 0u)
            rank = p;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (gid < P.n)
        keysOut[histScanned[d * P.numBlocks + groupId] + rank] = key;
}

// --------------------------------------------------------------------------
// Work queue: simdgroup-aggregated slot allocation — one fetch_add per
// simdgroup instead of per thread. Call under divergent control flow; simd
// ops operate on the active lanes.
// --------------------------------------------------------------------------

inline uint prim_queue_alloc(device atomic_uint* counter)
{
    uint lanePrefix = simd_prefix_exclusive_sum(1u);
    uint total = simd_sum(1u);
    uint base = 0u;
    if (simd_is_first())
        base = atomic_fetch_add_explicit(counter, total, memory_order_relaxed);
    base = simd_broadcast_first(base);
    return base + lanePrefix;
}

// Test kernel: threads whose value is odd enqueue it.
kernel void prim_queue_push_test(constant PrimParams& P    [[buffer(0)]],
                                 device const uint* in     [[buffer(1)]],
                                 device atomic_uint* count [[buffer(2)]],
                                 device uint* outItems     [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]])
{
    if (gid >= P.n)
        return;
    uint v = in[gid];
    if ((v & 1u) != 0u)
        outItems[prim_queue_alloc(count)] = v;
}
