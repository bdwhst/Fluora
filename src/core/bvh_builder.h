#pragma once
// Portable CPU build of the six-direction threaded BVH consumed by the device
// traversal in src/rhi/raytrace.metal (see RtBvhNode in accel_shared.h).
// Median split on the longest centroid axis; SAH is a quality upgrade for
// later, and parity with the CUDA renderer's SAH builder (bvh.cpp) is
// revisited in M4.
#include <cstdint>
#include <vector>
#include <simd/simd.h>

#include "accel_shared.h"

struct BvhBuildResult {
    std::vector<RtBvhNode> nodes;  // 6 direction-ordered copies, concatenated
    uint32_t numNodesPerDir = 0;
};

// Reorders `tris` in place (leaves reference contiguous ranges).
BvhBuildResult buildThreadedBvh6(const std::vector<simd_float3>& positions,
                                 std::vector<simd_uint4>& tris);
