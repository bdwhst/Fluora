#include "bvh_builder.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "host_math.h"

namespace {

// Split policy ported from the CUDA renderer's SAH builder (bvh.cpp) so both
// backends traverse the same tree in M4: 20-bucket surface-area heuristic on
// the largest centroid-extent axis, expanded per-primitive bboxes, leaves of
// at most 2 primitives (degenerate centroid extent stops earlier). The float
// expressions mirror bvh.cpp exactly — including the O(buckets^2) cost sweep —
// so near-tie split choices match.
constexpr float kBboxExpand = 0.0001f;      // BOUNDING_BOX_EXPAND
constexpr float kExtentEps = 0.00001f;      // EPSILON
constexpr int kSahBuckets = 20;             // SAH_BUCKET_SIZE
constexpr float kSahTraversalCost = 0.1f;   // SAH_RAY_BOX_INTERSECTION_COST
constexpr uint32_t kMaxLeafPrims = 2;       // MAX_NUM_PRIMS_IN_LEAF

struct Box {
    glm::vec3 mn { 1e38f, 1e38f, 1e38f };
    glm::vec3 mx { -1e38f, -1e38f, -1e38f };
    void grow(glm::vec3 p) { mn = glm::min(mn, p); mx = glm::max(mx, p); }
    void grow(const Box& b) { mn = glm::min(mn, b.mn); mx = glm::max(mx, b.mx); }
    glm::vec3 center() const { return (mn + mx) * 0.5f; }
    float area() const
    {
        glm::vec3 d = mx - mn;
        return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
    }
};

struct Prim {
    Box box;              // expanded per-triangle bbox
    gpu_uint4 tri;        // partitioned along with it, like bvh.cpp's Primitive
};

struct BuildNode {
    glm::vec3 bmin, bmax, centroid;
    int left = -1, right = -1;        // indices into the build-node pool
    uint32_t triStart = 0, triCount = 0;
    uint32_t subtreeSize = 1;
};

struct Builder {
    std::vector<Prim>& prims;         // reordered in place
    std::vector<BuildNode> nodes;

    int build(uint32_t start, uint32_t end)
    {
        Box bb, bCenter;
        for (uint32_t i = start; i < end; i++) {
            bb.grow(prims[i].box);
            bCenter.grow(prims[i].box.center());
        }
        uint32_t count = end - start;
        glm::vec3 cd = bCenter.mx - bCenter.mn;
        int axis = cd.x >= cd.y && cd.x >= cd.z ? 0 : (cd.y >= cd.z ? 1 : 2);

        BuildNode n;
        n.bmin = bb.mn;
        n.bmax = bb.mx;
        n.centroid = bb.center();
        if (count <= kMaxLeafPrims || cd[axis] <= kExtentEps) {
            n.triStart = start;
            n.triCount = count;
            nodes.push_back(n);
            return (int)nodes.size() - 1;
        }

        auto bucketOf = [&](const Prim& p) {
            int b = (int)((p.box.center()[axis] - bCenter.mn[axis]) / cd[axis]
                          * kSahBuckets);
            return b == kSahBuckets ? kSahBuckets - 1 : b;
        };
        struct Bucket {
            int cnt = 0;
            Box box;
        } buckets[kSahBuckets];
        for (uint32_t i = start; i < end; i++) {
            Bucket& b = buckets[bucketOf(prims[i])];
            b.cnt++;
            b.box.grow(prims[i].box);
        }
        float rootArea = bb.area();
        int minSplit = 0;
        float minCost = 1e38f;
        for (int i = 0; i < kSahBuckets - 1; i++) {
            Box b0, b1;
            int cnt0 = 0, cnt1 = 0;
            for (int j = 0; j <= i; j++) {
                cnt0 += buckets[j].cnt;
                b0.grow(buckets[j].box);
            }
            for (int j = i + 1; j < kSahBuckets; j++) {
                cnt1 += buckets[j].cnt;
                b1.grow(buckets[j].box);
            }
            float cost = kSahTraversalCost
                       + (cnt0 * b0.area() + cnt1 * b1.area()) / rootArea;
            if (cost < minCost) {
                minCost = cost;
                minSplit = i;
            }
        }
        // bvh.cpp splits whenever count > kMaxLeafPrims regardless of the leaf
        // cost; SAH only chooses where. Both sides are non-empty: the min and
        // max centroids land in buckets 0 and kSahBuckets-1.
        auto midIt = std::partition(prims.begin() + start, prims.begin() + end,
                                    [&](const Prim& p) {
                                        return bucketOf(p) <= minSplit;
                                    });
        uint32_t mid = (uint32_t)(midIt - prims.begin());

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
    // after the near subtree; missLink jumps over the whole subtree. Near/far
    // follows bvh.cpp: strict signed comparison of node-bbox centers.
    void emit(int ni, uint32_t missLink, int axis, bool negative, std::vector<RtBvhNode>& out)
    {
        const BuildNode& n = nodes[ni];
        RtBvhNode g = {};
        g.bmin = hostStore3(n.bmin);
        g.bmax = hostStore3(n.bmax);
        g.missLink = missLink;
        g.triStart = n.triStart;
        g.triCount = n.triCount;
        uint32_t idx = (uint32_t)(out.size() % nodes.size());  // index within this direction
        if (n.left < 0) {
            g.hitLink = missLink;  // leaf: after testing triangles, continue past
            out.push_back(g);
            return;
        }
        float sgn = negative ? -1.0f : 1.0f;
        int nearC = n.left, farC = n.right;
        if (!(nodes[n.left].centroid[axis] * sgn < nodes[n.right].centroid[axis] * sgn))
            std::swap(nearC, farC);
        g.hitLink = idx + 1;
        out.push_back(g);
        uint32_t farIdx = idx + 1 + nodes[nearC].subtreeSize;
        emit(nearC, farIdx, axis, negative, out);
        emit(farC, missLink, axis, negative, out);
    }
};

} // namespace

BvhBuildResult buildThreadedBvh6(const std::vector<gpu_storage3>& positions,
                                 std::vector<gpu_uint4>& tris)
{
    BvhBuildResult result;
    if (tris.empty())
        return result;
    std::vector<Prim> prims(tris.size());
    for (size_t i = 0; i < tris.size(); i++) {
        Prim p;
        for (int k = 0; k < 3; k++)
            p.box.grow(gpu_load3(positions[tris[i][k]]));
        p.box.mn -= kBboxExpand;
        p.box.mx += kBboxExpand;
        p.tri = tris[i];
        prims[i] = p;
    }
    Builder b{ prims, {} };
    b.nodes.reserve(tris.size() * 2);
    b.build(0, (uint32_t)tris.size());
    for (size_t i = 0; i < prims.size(); i++)
        tris[i] = prims[i].tri;
    result.numNodesPerDir = (uint32_t)b.nodes.size();
    result.nodes.reserve((size_t)result.numNodesPerDir * 6);
    for (int axis = 0; axis < 3; axis++)
        for (int neg = 0; neg < 2; neg++)
            b.emit(0, result.numNodesPerDir, axis, neg != 0, result.nodes);
    std::cout << "core: BVH " << result.numNodesPerDir << " nodes x6 over "
              << tris.size() << " tris\n";
    return result;
}
