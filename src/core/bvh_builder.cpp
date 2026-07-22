#include "bvh_builder.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

struct BuildNode {
    simd_float3 bmin, bmax, centroid;
    int left = -1, right = -1;        // indices into the build-node pool
    uint32_t triStart = 0, triCount = 0;
    uint32_t subtreeSize = 1;
};

struct Builder {
    const std::vector<simd_float3>& positions;
    std::vector<simd_uint4>& tris;    // reordered in place
    std::vector<BuildNode> nodes;

    simd_float3 centroidOf(const simd_uint4& t) const
    {
        return (positions[t.x] + positions[t.y] + positions[t.z]) * (1.0f / 3.0f);
    }

    int build(uint32_t start, uint32_t end)
    {
        BuildNode n;
        n.bmin = simd_make_float3(INFINITY, INFINITY, INFINITY);
        n.bmax = -n.bmin;
        simd_float3 cmin = n.bmin, cmax = n.bmax;
        for (uint32_t i = start; i < end; i++) {
            for (int k = 0; k < 3; k++) {
                simd_float3 p = positions[tris[i][k]];
                n.bmin = simd_min(n.bmin, p);
                n.bmax = simd_max(n.bmax, p);
            }
            simd_float3 c = centroidOf(tris[i]);
            cmin = simd_min(cmin, c);
            cmax = simd_max(cmax, c);
        }
        n.centroid = (n.bmin + n.bmax) * 0.5f;
        uint32_t count = end - start;
        simd_float3 ext = cmax - cmin;
        int axis = ext.x > ext.y ? (ext.x > ext.z ? 0 : 2) : (ext.y > ext.z ? 1 : 2);
        if (count <= 4 || ext[axis] <= 0.0f) {
            n.triStart = start;
            n.triCount = count;
            nodes.push_back(n);
            return (int)nodes.size() - 1;
        }
        uint32_t mid = start + count / 2;
        std::nth_element(tris.begin() + start, tris.begin() + mid, tris.begin() + end,
                         [&](const simd_uint4& a, const simd_uint4& b) {
                             return centroidOf(a)[axis] < centroidOf(b)[axis];
                         });
        int idx = (int)nodes.size();
        nodes.push_back(n);
        int l = build(start, mid);
        int r = build(mid, end);
        nodes[idx].left = l;
        nodes[idx].right = r;
        nodes[idx].subtreeSize = 1 + nodes[l].subtreeSize + nodes[r].subtreeSize;
        return idx;
    }

    // Threaded DFS emission for one direction: near child at idx+1, far child
    // after the near subtree; missLink jumps over the whole subtree.
    void emit(int ni, uint32_t missLink, int axis, bool negative, std::vector<RtBvhNode>& out)
    {
        const BuildNode& n = nodes[ni];
        RtBvhNode g = {};
        g.bmin = n.bmin;
        g.bmax = n.bmax;
        g.missLink = missLink;
        g.triStart = n.triStart;
        g.triCount = n.triCount;
        uint32_t idx = (uint32_t)(out.size() % nodes.size());  // index within this direction
        if (n.left < 0) {
            g.hitLink = missLink;  // leaf: after testing triangles, continue past
            out.push_back(g);
            return;
        }
        int nearC = n.left, farC = n.right;
        bool leftNear = nodes[n.left].centroid[axis] <= nodes[n.right].centroid[axis];
        if (leftNear == negative)
            std::swap(nearC, farC);
        g.hitLink = idx + 1;
        out.push_back(g);
        uint32_t farIdx = idx + 1 + nodes[nearC].subtreeSize;
        emit(nearC, farIdx, axis, negative, out);
        emit(farC, missLink, axis, negative, out);
    }
};

} // namespace

BvhBuildResult buildThreadedBvh6(const std::vector<simd_float3>& positions,
                                 std::vector<simd_uint4>& tris)
{
    BvhBuildResult result;
    if (tris.empty())
        return result;
    Builder b{ positions, tris, {} };
    b.nodes.reserve(tris.size() * 2);
    b.build(0, (uint32_t)tris.size());
    result.numNodesPerDir = (uint32_t)b.nodes.size();
    result.nodes.reserve((size_t)result.numNodesPerDir * 6);
    for (int axis = 0; axis < 3; axis++)
        for (int neg = 0; neg < 2; neg++)
            b.emit(0, result.numNodesPerDir, axis, neg != 0, result.nodes);
    std::cout << "core: BVH " << result.numNodesPerDir << " nodes x6 over "
              << tris.size() << " tris\n";
    return result;
}
