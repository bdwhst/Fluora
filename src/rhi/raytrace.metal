// Device-side ray tracing seam (design doc §5): renderer kernels call
// rt_closest_hit() and stay ignorant of the traversal technique. This
// implementation walks the six-direction threaded BVH built by
// src/core/bvh_builder.cpp (mini analog of the CUDA MTBVH in
// intersections.cu); a Metal hardware-RT implementation with the same
// signature is the M5 fast path.
//
// Concatenated after core/accel_shared.h (RtBvhNode); do not #include.
using namespace metal;

struct RtHit {
    float t;           // in: max distance; out: hit distance
    float3 n;          // geometric normal (unoriented)
    float2 bary;       // barycentrics (weights of v1, v2)
    uint triIdx;       // index into the (reordered) triangle array
    uint userData;     // triangle uint4.w (renderer stores a material id)
    bool hit;
};

inline bool rt_intersect_triangle(float3 ro, float3 rd, float3 v0, float3 v1, float3 v2,
                                  thread float& tOut, thread float2& baryOut)
{
    // Moller-Trumbore, no backface culling (dielectrics need both sides)
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 pv = cross(rd, e2);
    float det = dot(e1, pv);
    if (fabs(det) < 1e-9f)
        return false;
    float invDet = 1.0f / det;
    float3 tv = ro - v0;
    float u = dot(tv, pv) * invDet;
    if (u < 0.0f || u > 1.0f)
        return false;
    float3 qv = cross(tv, e1);
    float v = dot(rd, qv) * invDet;
    if (v < 0.0f || u + v > 1.0f)
        return false;
    float t = dot(e2, qv) * invDet;
    if (t < 1e-5f)
        return false;
    tOut = t;
    baryOut = float2(u, v);
    return true;
}

inline bool rt_aabb_hit(float3 ro, float3 invD, float3 bmin, float3 bmax, float tMax)
{
    float3 t0 = (bmin - ro) * invD;
    float3 t1 = (bmax - ro) * invD;
    float3 tS = min(t0, t1);
    float3 tB = max(t0, t1);
    float tmin = max(max(tS.x, tS.y), tS.z);
    float tmax = min(min(tB.x, tB.y), tB.z);
    return tmax >= max(tmin, 0.0f) && tmin < tMax;
}

// Stackless traversal: pick the direction-ordered node array by the ray's
// dominant axis/sign, then follow hit/miss links — no stack, front-to-back-ish.
// hit.t must be initialized to the current closest distance (or INFINITY).
inline void rt_closest_hit(float3 ro, float3 rd,
                           device const RtBvhNode* nodes, uint numNodes,
                           device const uint4* tris, device const float3* positions,
                           thread RtHit& hit)
{
    hit.hit = false;
    if (numNodes == 0)
        return;
    float3 a = abs(rd);
    uint axis = a.x > a.y ? (a.x > a.z ? 0u : 2u) : (a.y > a.z ? 1u : 2u);
    uint dirIdx = axis * 2u + (rd[axis] < 0.0f ? 1u : 0u);
    device const RtBvhNode* base = nodes + dirIdx * numNodes;
    float3 invD = 1.0f / rd;
    int bestTri = -1;
    float2 bestBary = float2(0.0f);
    uint curr = 0;
    while (curr < numNodes) {
        device const RtBvhNode& node = base[curr];
        if (rt_aabb_hit(ro, invD, node.bmin, node.bmax, hit.t)) {
            for (uint i = 0; i < node.triCount; i++) {
                uint4 tri = tris[node.triStart + i];
                float t;
                float2 bary;
                if (rt_intersect_triangle(ro, rd, positions[tri.x], positions[tri.y],
                                          positions[tri.z], t, bary) && t < hit.t) {
                    hit.t = t;
                    bestTri = (int)(node.triStart + i);
                    bestBary = bary;
                }
            }
            curr = node.hitLink;
        } else {
            curr = node.missLink;
        }
    }
    if (bestTri >= 0) {
        uint4 tri = tris[bestTri];
        hit.n = normalize(cross(positions[tri.y] - positions[tri.x],
                                positions[tri.z] - positions[tri.x]));
        hit.bary = bestBary;
        hit.triIdx = (uint)bestTri;
        hit.userData = tri.w;
        hit.hit = true;
    }
}

// Vertex-attribute interpolation for a hit; the attribute arrays share the
// triangle's vertex indices (unified vertices, core/mesh_loader).

// Zero-length vertex normals mean "not provided" — fall back to geometric.
inline float3 rt_shading_normal(device const uint4* tris, device const float3* normals,
                                thread const RtHit& hit)
{
    uint4 tri = tris[hit.triIdx];
    float3 n = normals[tri.x] * (1.0f - hit.bary.x - hit.bary.y)
             + normals[tri.y] * hit.bary.x
             + normals[tri.z] * hit.bary.y;
    return dot(n, n) > 1e-12f ? normalize(n) : hit.n;
}

inline float2 rt_interp_uv(device const uint4* tris, device const float2* uvs,
                           thread const RtHit& hit)
{
    uint4 tri = tris[hit.triIdx];
    return uvs[tri.x] * (1.0f - hit.bary.x - hit.bary.y)
         + uvs[tri.y] * hit.bary.x
         + uvs[tri.z] * hit.bary.y;
}
