#pragma once
// Portable CPU build of the six-direction threaded BVH consumed by the device
// traversal in src/rhi/raytrace_gpu.h (see RtBvhNode in accel_shared.h).
// Bucketed SAH split ported from the CUDA renderer's builder (bvh.cpp, same
// constants and float expressions); M4 points the CUDA backend at this builder
// for node-for-node traversal parity and retires bvh.cpp's copy.
#include <cstdint>
#include <vector>

#include "../rhi/gpu_portable.h"
#include "accel_shared.h"

struct BvhBuildResult {
    std::vector<RtBvhNode> nodes;  // 6 direction-ordered copies, concatenated
    uint32_t numNodesPerDir = 0;
};

// Reorders `tris` in place (leaves reference contiguous ranges).
BvhBuildResult buildThreadedBvh6(const std::vector<gpu_storage3>& positions,
                                 std::vector<gpu_uint4>& tris);
