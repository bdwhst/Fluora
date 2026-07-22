// FluoraMini path tracing kernels (docs/metal-rhi-design.md M1+M2).
// Two execution modes share every helper below, so their images are bitwise
// identical:
//  - megakernel (M1): one dispatch per sample, full path per thread
//  - wavefront (M2): raygen -> per-bounce [intersect -> shade] stages with
//    atomic work queues and indirect dispatch; terminated paths simply are
//    not re-enqueued (no compaction pass)
//
// Compiled after mini_shared.h and primitives_shared.h/primitives.metal are
// textually prepended (prim_queue_alloc comes from there); do not #include.
using namespace metal;

struct MiniRng {
    uint state;
};

inline float mini_rand(thread MiniRng& r)
{
    // PCG output permutation on an LCG state
    r.state = r.state * 747796405u + 2891336453u;
    uint w = ((r.state >> ((r.state >> 28u) + 4u)) ^ r.state) * 277803737u;
    w = (w >> 22u) ^ w;
    return (float)w * (1.0f / 4294967296.0f);
}

// Slab test against the unit cube [-0.5, 0.5]^3 in object space. Ray direction
// is unnormalized (object-space transform of a world ray), so t is measured in
// world units and comparable across objects.
inline bool intersectUnitCube(float3 ro, float3 rd, thread float& tOut, thread float3& nOut)
{
    float3 invD = 1.0f / rd;
    float3 tA = (float3(-0.5f) - ro) * invD;
    float3 tB = (float3(0.5f) - ro) * invD;
    float3 tSmall = min(tA, tB);
    float3 tBig = max(tA, tB);
    float tmin = max(max(tSmall.x, tSmall.y), tSmall.z);
    float tmax = min(min(tBig.x, tBig.y), tBig.z);
    if (tmax < tmin || tmax < 1e-6f)
        return false;
    float t = tmin > 1e-6f ? tmin : tmax;
    float3 p = ro + rd * t;
    float3 ap = abs(p);
    float m = max(max(ap.x, ap.y), ap.z);
    float3 n = float3(0.0f);
    if (m == ap.x)      n.x = p.x > 0.0f ? 1.0f : -1.0f;
    else if (m == ap.y) n.y = p.y > 0.0f ? 1.0f : -1.0f;
    else                n.z = p.z > 0.0f ? 1.0f : -1.0f;
    tOut = t;
    nOut = n;
    return true;
}

inline bool intersectUnitSphere(float3 ro, float3 rd, thread float& tOut, thread float3& nOut)
{
    float a = dot(rd, rd);
    float b = 2.0f * dot(ro, rd);
    float c = dot(ro, ro) - 0.25f;
    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f)
        return false;
    float sq = sqrt(disc);
    float t0 = (-b - sq) / (2.0f * a);
    float t1 = (-b + sq) / (2.0f * a);
    float t = t0 > 1e-6f ? t0 : (t1 > 1e-6f ? t1 : -1.0f);
    if (t < 0.0f)
        return false;
    tOut = t;
    nOut = normalize(ro + rd * t);
    return true;
}

struct MiniHit {
    float t;
    float3 n;      // world-space geometric normal
    int objIdx;    // >=0 analytic object, -2 mesh triangle, -1 miss
    uint matId;
};

// Analytic scaffolding objects tested brute-force; mesh geometry goes through
// the rt_closest_hit seam (src/rhi/raytrace.metal).
inline bool sceneIntersect(float3 ro, float3 rd,
                           device const MiniObject* objects, uint numObjects,
                           device const RtBvhNode* nodes, uint numNodes,
                           device const uint4* tris, device const float3* positions,
                           thread MiniHit& hit)
{
    hit.t = INFINITY;
    hit.objIdx = -1;
    hit.matId = 0;
    for (uint i = 0; i < numObjects; i++) {
        device const MiniObject& obj = objects[i];
        float3 roLocal = (obj.invTransform * float4(ro, 1.0f)).xyz;
        float3 rdLocal = (obj.invTransform * float4(rd, 0.0f)).xyz;
        float t;
        float3 nLocal;
        bool ok = (obj.geomType == MINI_GEOM_CUBE)
            ? intersectUnitCube(roLocal, rdLocal, t, nLocal)
            : intersectUnitSphere(roLocal, rdLocal, t, nLocal);
        if (ok && t < hit.t) {
            hit.t = t;
            hit.n = normalize((obj.invTranspose * float4(nLocal, 0.0f)).xyz);
            hit.objIdx = (int)i;
            hit.matId = (uint)obj.materialId;
        }
    }
    RtHit rhit;
    rhit.t = hit.t;
    rt_closest_hit(ro, rd, nodes, numNodes, tris, positions, rhit);
    if (rhit.hit) {
        hit.t = rhit.t;
        hit.n = rhit.n;
        hit.matId = rhit.userData;
        hit.objIdx = -2;
    }
    return hit.objIdx != -1;
}

// Same jittered pinhole formula as generateRayFromCamera in pathtrace.cu.
inline void generateCameraRay(constant MiniParams& P, uint2 gid, thread MiniRng& rng,
                              thread float3& ro, thread float3& rd)
{
    float jx = mini_rand(rng) - 0.5f;
    float jy = mini_rand(rng) - 0.5f;
    rd = normalize(P.camView
        - P.camRight * P.pixelLenX * ((float)gid.x - (float)P.width * 0.5f + jx)
        - P.camUp    * P.pixelLenY * ((float)gid.y - (float)P.height * 0.5f + jy));
    ro = P.camPos;
}

// Draws this material's uniforms and scatters via the core BSDFs
// (core/bsdf_shared.h). Shared by the megakernel switch and the specialized
// shade kernel, so both modes consume identical RNG streams and math.
// `type` is mat.type in the megakernel (dynamic branch) and a function
// constant in wf_shade (branch folds at pipeline creation).
// Returns false when the sample is absorbed.
inline bool miniScatter(uint type, device const MiniMaterial& mat, float3 p, float3 n,
                        thread float3& ro, thread float3& rd,
                        thread float3& throughput, thread MiniRng& rng)
{
    float3 nF = dot(n, rd) < 0.0f ? n : -n;
    bool alive;
    if (type == MINI_MAT_DIFFUSE) {
        float u1 = mini_rand(rng);
        float u2 = mini_rand(rng);
        alive = bsdf_sample_lambert(float3(mat.rgb), nF, u1, u2, rd, throughput);
    } else if (type == MINI_MAT_CONDUCTOR) {
        float u1 = mini_rand(rng);
        float u2 = mini_rand(rng);
        alive = bsdf_sample_conductor(float3(mat.rgb), mat.roughness, nF, u1, u2, rd, throughput);
    } else {  // MINI_MAT_GLASS
        float u = mini_rand(rng);
        alive = bsdf_sample_dielectric(float3(mat.rgb), mat.ior, n, u, rd, throughput);
    }
    if (!alive)
        return false;
    ro = p + nF * (dot(rd, nF) >= 0.0f ? 1e-4f : -1e-4f);
    return true;
}

// ==========================================================================
// Megakernel mode (M1)
// ==========================================================================

kernel void pathtraceKernel(constant MiniParams& P                [[buffer(0)]],
                            device float4* accum                  [[buffer(1)]],
                            device const MiniMaterial* materials  [[buffer(2)]],
                            device const MiniObject* objects      [[buffer(3)]],
                            device const RtBvhNode* bvhNodes    [[buffer(4)]],
                            device const uint4* tris              [[buffer(5)]],
                            device const float3* positions        [[buffer(6)]],
                            uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= P.width || gid.y >= P.height)
        return;
    uint idx = gid.y * P.width + gid.x;

    MiniRng rng;
    rng.state = idx * 9781u + P.iter * 26699u + 1u;
    mini_rand(rng);  // decorrelate the low-entropy seed
    mini_rand(rng);

    float3 ro, rd;
    generateCameraRay(P, gid, rng, ro, rd);

    float3 L = float3(0.0f);
    float3 throughput = float3(1.0f);

    for (uint depth = 0; depth < P.maxDepth; depth++) {
        MiniHit hit;
        if (!sceneIntersect(ro, rd, objects, P.numObjects, bvhNodes, P.bvhNumNodes,
                            tris, positions, hit))
            break;  // black environment, like the CUDA renderer without a skybox
        device const MiniMaterial& mat = materials[hit.matId];
        if (mat.type == MINI_MAT_EMITTING) {
            L += throughput * float3(mat.rgb) * mat.emittance;
            break;
        }
        if (!miniScatter(mat.type, mat, ro + rd * hit.t, hit.n, ro, rd, throughput, rng))
            break;
    }

    if (all(isfinite(L)))
        accum[idx] += float4(L, 1.0f);
}

// ==========================================================================
// Wavefront mode (M2)
// ==========================================================================

struct WfPath {
    packed_float3 origin;     float t;
    packed_float3 dir;        uint pixel;
    packed_float3 throughput; uint rng;
    packed_float3 normal;     uint depth;
    uint matId; uint pad0; uint pad1; uint pad2;
};
static_assert(sizeof(WfPath) == WF_PATHSTATE_SIZE, "host allocates queues with this stride");

kernel void wf_raygen(constant MiniParams& P     [[buffer(0)]],
                      device WfPath* rays        [[buffer(1)]],
                      device atomic_uint* counts [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= P.width || gid.y >= P.height)
        return;
    uint idx = gid.y * P.width + gid.x;

    MiniRng rng;
    rng.state = idx * 9781u + P.iter * 26699u + 1u;
    mini_rand(rng);
    mini_rand(rng);

    float3 ro, rd;
    generateCameraRay(P, gid, rng, ro, rd);

    WfPath path;
    path.origin = ro;
    path.dir = rd;
    path.throughput = float3(1.0f);
    path.normal = float3(0.0f);
    path.t = 0.0f;
    path.pixel = idx;
    path.rng = rng.state;
    path.depth = 0;
    path.matId = 0;
    rays[idx] = path;

    if (idx == 0) {
        // Shade counters are zeroed by wf_prep_intersect each bounce.
        atomic_store_explicit(&counts[WF_COUNT_RAY_A], P.width * P.height, memory_order_relaxed);
        atomic_store_explicit(&counts[WF_COUNT_RAY_B], 0u, memory_order_relaxed);
    }
}

// Single-thread dispatches turning GPU-written queue counts into indirect
// threadgroup args, keeping the bounce loop free of CPU readbacks.
kernel void wf_prep_intersect(constant WfCtl& C          [[buffer(0)]],
                              device atomic_uint* counts [[buffer(1)]],
                              device uint* args          [[buffer(2)]])
{
    uint c = atomic_load_explicit(&counts[C.srcCounter], memory_order_relaxed);
    args[WF_ARG_INTERSECT * 4 + 0] = (c + PRIM_TILE - 1u) / PRIM_TILE;
    args[WF_ARG_INTERSECT * 4 + 1] = 1u;
    args[WF_ARG_INTERSECT * 4 + 2] = 1u;
    atomic_store_explicit(&counts[WF_COUNT_SHADE_DIFFUSE], 0u, memory_order_relaxed);
    atomic_store_explicit(&counts[WF_COUNT_SHADE_CONDUCTOR], 0u, memory_order_relaxed);
    atomic_store_explicit(&counts[WF_COUNT_SHADE_GLASS], 0u, memory_order_relaxed);
}

kernel void wf_prep_shade(constant WfCtl& C          [[buffer(0)]],
                          device atomic_uint* counts [[buffer(1)]],
                          device uint* args          [[buffer(2)]])
{
    constexpr uint queues[3] = { WF_COUNT_SHADE_DIFFUSE, WF_COUNT_SHADE_CONDUCTOR,
                                 WF_COUNT_SHADE_GLASS };
    constexpr uint slots[3] = { WF_ARG_DIFFUSE, WF_ARG_CONDUCTOR, WF_ARG_GLASS };
    for (uint i = 0; i < 3; i++) {
        uint c = atomic_load_explicit(&counts[queues[i]], memory_order_relaxed);
        args[slots[i] * 4 + 0] = (c + PRIM_TILE - 1u) / PRIM_TILE;
        args[slots[i] * 4 + 1] = 1u;
        args[slots[i] * 4 + 2] = 1u;
    }
    atomic_store_explicit(&counts[C.zeroCounter], 0u, memory_order_relaxed);
}

// Intersect routes surviving paths into their material type's shade queue
// (tier-1 material dispatch: the queue decides which BSDF code runs, not a
// per-thread branch). Emissive hits are resolved here.
kernel void wf_intersect(constant WfCtl& C                 [[buffer(0)]],
                         device atomic_uint* counts        [[buffer(1)]],
                         device const WfPath* raysIn       [[buffer(2)]],
                         device const MiniObject* objects  [[buffer(3)]],
                         device const RtBvhNode* bvhNodes  [[buffer(4)]],
                         device const uint4* tris          [[buffer(5)]],
                         device const float3* positions    [[buffer(6)]],
                         device const MiniMaterial* materials [[buffer(7)]],
                         device float4* accum              [[buffer(8)]],
                         device WfPath* qDiffuse           [[buffer(9)]],
                         device WfPath* qConductor         [[buffer(10)]],
                         device WfPath* qGlass             [[buffer(11)]],
                         uint tid [[thread_position_in_grid]])
{
    if (tid >= atomic_load_explicit(&counts[C.srcCounter], memory_order_relaxed))
        return;
    WfPath path = raysIn[tid];
    MiniHit hit;
    if (!sceneIntersect(float3(path.origin), float3(path.dir), objects, C.numObjects,
                        bvhNodes, C.bvhNumNodes, tris, positions, hit))
        return;  // escaped: contributes nothing, simply not re-enqueued

    device const MiniMaterial& mat = materials[hit.matId];
    if (mat.type == MINI_MAT_EMITTING) {
        // One path per pixel per sample, so this write does not race.
        float3 L = float3(path.throughput) * float3(mat.rgb) * mat.emittance;
        if (all(isfinite(L)))
            accum[path.pixel] += float4(L, 1.0f);
        return;
    }

    path.t = hit.t;
    path.normal = hit.n;
    path.matId = hit.matId;
    if (mat.type == MINI_MAT_DIFFUSE)
        qDiffuse[prim_queue_alloc(&counts[WF_COUNT_SHADE_DIFFUSE])] = path;
    else if (mat.type == MINI_MAT_CONDUCTOR)
        qConductor[prim_queue_alloc(&counts[WF_COUNT_SHADE_CONDUCTOR])] = path;
    else
        qGlass[prim_queue_alloc(&counts[WF_COUNT_SHADE_GLASS])] = path;
}

// One shade kernel, specialized per material type at pipeline creation
// (rhi::SpecConstant -> MTLFunctionConstantValues): the miniScatter branch
// folds away, and the queue guarantees the specialization matches every path
// in it — divergence-free shading with a single source of truth.
constant uint kShadeMatType [[function_constant(0)]];

kernel void wf_shade(constant WfCtl& C                    [[buffer(0)]],
                     device atomic_uint* counts           [[buffer(1)]],
                     device const WfPath* queue           [[buffer(2)]],
                     device WfPath* raysOut               [[buffer(3)]],
                     device const MiniMaterial* materials [[buffer(4)]],
                     uint tid [[thread_position_in_grid]])
{
    if (tid >= atomic_load_explicit(&counts[C.srcCounter], memory_order_relaxed))
        return;
    WfPath path = queue[tid];
    if (path.depth + 1 >= C.maxDepth)
        return;
    MiniRng rng;
    rng.state = path.rng;
    float3 ro = float3(path.origin);
    float3 rd = float3(path.dir);
    float3 throughput = float3(path.throughput);
    if (!miniScatter(kShadeMatType, materials[path.matId], ro + rd * path.t,
                     float3(path.normal), ro, rd, throughput, rng))
        return;
    path.origin = ro;
    path.dir = rd;
    path.throughput = throughput;
    path.rng = rng.state;
    path.depth++;
    raysOut[prim_queue_alloc(&counts[C.dstCounter])] = path;
}

// ==========================================================================
// Preview
// ==========================================================================

// Tonemaps the accumulator into the RHI present target (RGBA8) each iteration.
// P.iter carries the number of completed samples. Mirrors x like saveImage()
// (the quirk all saved renders share), so the window shows exactly what the
// PNG will contain.
kernel void present_tonemap(constant MiniParams& P     [[buffer(0)]],
                            device const float4* accum [[buffer(1)]],
                            device uchar4* out         [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= P.width || gid.y >= P.height)
        return;
    float3 c = tonemap_aces(accum[gid.y * P.width + gid.x].xyz / (float)P.iter);
    out[gid.y * P.width + (P.width - 1u - gid.x)] = uchar4(uchar3(c * 255.0f + 0.5f), 255);
}
