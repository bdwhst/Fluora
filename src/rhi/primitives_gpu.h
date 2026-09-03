#ifndef RHI_PRIMITIVES_GPU_H
#define RHI_PRIMITIVES_GPU_H
// Parallel primitives (design doc M2): reduce, exclusive scan, compact
// scatter, LSD radix sort passes, and the wave-aggregated work-queue push.
// Single-source via the gpu_portable shim (docs/portable-device-code.md):
// MSL today, CUDA in M4 through the same wave/threadgroup shims (a primitive
// may fall back to a CUB-backed same-named kernel per invariant I-2 only if a
// wave shim proves divergent — measured exception, never the default).
//
// Assumes 32-wide waves (Apple simdgroups, NVIDIA warps) and PRIM_TILE (256)
// threads per threadgroup, one element per thread.

#ifndef __METAL_VERSION__
#include "gpu_portable.h"
#include "primitives_shared.h"
#endif

// Exclusive prefix sum of v across the threadgroup, in thread-index order.
// sgScratch must hold PRIM_TILE/32 entries of threadgroup memory. Contains
// barriers: call from uniform control flow only, and barrier before reusing
// sgScratch.
GPU_FN inline uint tg_exclusive_scan(uint v, uint lid, uint sgId, uint numSg,
                              GPU_SHARED_SPACE uint* sgScratch)
{
    gpu_wave_t w = gpu_wave_active();
    uint lanePrefix = gpu_wave_prefix_sum(w, v);
    uint sgTotal = gpu_wave_sum(w, v);
    if (gpu_wave_is_first(w))
        sgScratch[sgId] = sgTotal;
    gpu_barrier();
    if (sgId == 0) {
        uint x = lid < numSg ? sgScratch[lid] : 0u;
        uint p = gpu_wave_prefix_sum(w, x);
        if (lid < numSg)
            sgScratch[lid] = p;
    }
    gpu_barrier();
    return lanePrefix + sgScratch[sgId];
}

// Kernels (see the note at the work-queue test kernel below).
#ifndef GPU_PRIMITIVES_HELPERS_ONLY
// --------------------------------------------------------------------------
// Reduce
// --------------------------------------------------------------------------

GPU_KERNEL(prim_reduce_sum)(GPU_KERNEL_PARAMS(PrimParams, P)
    GPU_BUFFER(const uint, in, 1)
    GPU_BUFFER(gpu_atomic_uint, out, 2)
    GPU_TID_FULL)
{
    uint gid = GPU_GLOBAL_ID_X;
    uint lid = GPU_LOCAL_ID;
    uint sgId = GPU_WAVE_INDEX;
    uint numSg = GPU_NUM_WAVES;
    GPU_SHARED uint sgSums[PRIM_TILE / 32u];
    gpu_wave_t w = gpu_wave_active();
    uint v = gid < P.n ? in[gid] : 0u;
    uint s = gpu_wave_sum(w, v);
    if (gpu_wave_is_first(w))
        sgSums[sgId] = s;
    gpu_barrier();
    if (lid == 0) {
        uint total = 0;
        for (uint i = 0; i < numSg; i++)
            total += sgSums[i];
        gpu_atomic_fetch_add(out, total);
    }
}

// --------------------------------------------------------------------------
// Exclusive scan: per-block scan + block sums, then host recursion over the
// block sums, then offset add. (Multi-dispatch reduce-then-scan; single-pass
// decoupled-lookback needs forward-progress guarantees Apple GPUs don't make.)
// --------------------------------------------------------------------------

GPU_KERNEL(prim_scan_block)(GPU_KERNEL_PARAMS(PrimParams, P)
    GPU_BUFFER(const uint, in, 1)
    GPU_BUFFER(uint, out, 2)
    GPU_BUFFER(uint, blockSums, 3)
    GPU_TID_FULL)
{
    uint gid = GPU_GLOBAL_ID_X;
    uint lid = GPU_LOCAL_ID;
    GPU_SHARED uint sgScratch[PRIM_TILE / 32u];
    uint v = gid < P.n ? in[gid] : 0u;
    uint p = tg_exclusive_scan(v, lid, GPU_WAVE_INDEX, GPU_NUM_WAVES, sgScratch);
    if (gid < P.n)
        out[gid] = p;
    if (lid == PRIM_TILE - 1u)
        blockSums[GPU_GROUP_ID] = p + v;
}

GPU_KERNEL(prim_scan_add_offsets)(GPU_KERNEL_PARAMS(PrimParams, P)
    GPU_BUFFER(uint, out, 1)
    GPU_BUFFER(const uint, scannedBlockSums, 2)
    GPU_TID_FULL)
{
    uint gid = GPU_GLOBAL_ID_X;
    if (gid < P.n)
        out[gid] += scannedBlockSums[GPU_GROUP_ID];
}

// --------------------------------------------------------------------------
// Compact: out[scannedFlags[i]] = in[i] where flags[i] != 0. The caller scans
// the flags first; count = scannedFlags[n-1] + flags[n-1].
// --------------------------------------------------------------------------

GPU_KERNEL(prim_compact_scatter)(GPU_KERNEL_PARAMS(PrimParams, P)
    GPU_BUFFER(const uint, in, 1)
    GPU_BUFFER(const uint, flags, 2)
    GPU_BUFFER(const uint, scannedFlags, 3)
    GPU_BUFFER(uint, out, 4)
    GPU_TID_1D)
{
    uint gid = GPU_GLOBAL_ID_X;
    if (gid < P.n && flags[gid] != 0u)
        out[scannedFlags[gid]] = in[gid];
}

// --------------------------------------------------------------------------
// LSD radix sort, PRIM_RADIX_BITS per pass. Digit-major histogram layout
// (hist[digit * numBlocks + block]) so one exclusive scan of the whole
// histogram yields stable global scatter bases.
// --------------------------------------------------------------------------

GPU_KERNEL(prim_radix_histogram)(GPU_KERNEL_PARAMS(PrimParams, P)
    GPU_BUFFER(const uint, keys, 1)
    GPU_BUFFER(uint, hist, 2)
    GPU_TID_FULL)
{
    uint gid = GPU_GLOBAL_ID_X;
    uint lid = GPU_LOCAL_ID;
    GPU_SHARED gpu_atomic_uint local[PRIM_RADIX_DIGITS];
    if (lid < PRIM_RADIX_DIGITS)
        gpu_atomic_store(&local[lid], 0u);
    gpu_barrier();
    if (gid < P.n) {
        uint d = (keys[gid] >> P.shift) & (PRIM_RADIX_DIGITS - 1u);
        gpu_atomic_fetch_add(&local[d], 1u);
    }
    gpu_barrier();
    if (lid < PRIM_RADIX_DIGITS)
        hist[lid * P.numBlocks + GPU_GROUP_ID] = gpu_atomic_load(&local[lid]);
}

GPU_KERNEL(prim_radix_scatter)(GPU_KERNEL_PARAMS(PrimParams, P)
    GPU_BUFFER(const uint, keysIn, 1)
    GPU_BUFFER(uint, keysOut, 2)
    GPU_BUFFER(const uint, histScanned, 3)
    GPU_TID_FULL)
{
    uint gid = GPU_GLOBAL_ID_X;
    uint lid = GPU_LOCAL_ID;
    GPU_SHARED uint sgScratch[PRIM_TILE / 32u];
    uint key = gid < P.n ? keysIn[gid] : 0u;
    uint d = (key >> P.shift) & (PRIM_RADIX_DIGITS - 1u);
    // Stable rank among this block's earlier same-digit elements: one
    // threadgroup scan per digit value (correctness-first; a ballot-based
    // ranking can replace this if sort ever becomes hot).
    uint rank = 0u;
    for (uint b = 0; b < PRIM_RADIX_DIGITS; b++) {
        uint flag = (gid < P.n && d == b) ? 1u : 0u;
        uint p = tg_exclusive_scan(flag, lid, GPU_WAVE_INDEX, GPU_NUM_WAVES, sgScratch);
        if (flag != 0u)
            rank = p;
        gpu_barrier();
    }
    if (gid < P.n)
        keysOut[histScanned[d * P.numBlocks + GPU_GROUP_ID] + rank] = key;
}
#endif  // GPU_PRIMITIVES_HELPERS_ONLY

// --------------------------------------------------------------------------
// Work queue: wave-aggregated slot allocation — one fetch_add per wave
// instead of per thread. Call under divergent control flow; the wave ops
// operate on the active lanes.
// --------------------------------------------------------------------------

GPU_FN inline uint prim_queue_alloc(GPU_DEVICE gpu_atomic_uint* counter)
{
    // One handle for the whole sequence: the broadcast after the divergent
    // leader branch must address the same lane set that computed the ranks.
    gpu_wave_t w = gpu_wave_active();
    uint lanePrefix = gpu_wave_prefix_sum(w, 1u);
    uint total = gpu_wave_sum(w, 1u);
    uint base = 0u;
    if (gpu_wave_is_first(w))
        base = gpu_atomic_fetch_add(counter, total);
    base = gpu_wave_broadcast_first(w, base);
    return base + lanePrefix;
}

// Kernels are compiled only in the TU that registers them (rhi_cuda.cu on
// CUDA); renderer files that need just the helpers above define
// GPU_PRIMITIVES_HELPERS_ONLY before including this header.
#ifndef GPU_PRIMITIVES_HELPERS_ONLY
// Test kernel: threads whose value is odd enqueue it.
GPU_KERNEL(prim_queue_push_test)(GPU_KERNEL_PARAMS(PrimParams, P)
    GPU_BUFFER(const uint, in, 1)
    GPU_BUFFER(gpu_atomic_uint, count, 2)
    GPU_BUFFER(uint, outItems, 3)
    GPU_TID_1D)
{
    uint gid = GPU_GLOBAL_ID_X;
    if (gid >= P.n)
        return;
    uint v = in[gid];
    if ((v & 1u) != 0u)
        outItems[prim_queue_alloc(count)] = v;
}
#endif  // GPU_PRIMITIVES_HELPERS_ONLY

#endif // RHI_PRIMITIVES_GPU_H
