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
    int objIdx;
};

inline bool closestHit(float3 ro, float3 rd, device const MiniObject* objects,
                       uint numObjects, thread MiniHit& hit)
{
    hit.t = INFINITY;
    hit.objIdx = -1;
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
        }
    }
    return hit.objIdx >= 0;
}

inline float3 cosineSampleHemisphere(float3 n, thread MiniRng& rng)
{
    float u1 = mini_rand(rng);
    float u2 = mini_rand(rng);
    float r = sqrt(u1);
    float phi = 2.0f * M_PI_F * u2;
    // orthonormal basis around n
    float3 t = abs(n.x) > 0.9f ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 b1 = normalize(cross(n, t));
    float3 b2 = cross(n, b1);
    return normalize(b1 * (r * cos(phi)) + b2 * (r * sin(phi)) + n * sqrt(max(0.0f, 1.0f - u1)));
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

// Non-emissive scatter (emissive termination is the caller's job, since it
// touches the accumulator). Updates ro/rd/throughput in place.
inline void scatterMaterial(device const MiniMaterial& mat, float3 p, float3 n,
                            thread float3& ro, thread float3& rd,
                            thread float3& throughput, thread MiniRng& rng)
{
    float3 nFacing = dot(n, rd) < 0.0f ? n : -n;
    if (mat.type == MINI_MAT_DIFFUSE) {
        throughput *= float3(mat.rgb);
        rd = cosineSampleHemisphere(nFacing, rng);
        ro = p + nFacing * 1e-4f;
    }
    else if (mat.type == MINI_MAT_MIRROR) {
        throughput *= float3(mat.rgb);
        rd = reflect(rd, nFacing);
        ro = p + nFacing * 1e-4f;
    }
    else {  // MINI_MAT_GLASS — RTIOW-grade dielectric with Schlick fresnel
        bool entering = dot(n, rd) < 0.0f;
        float eta = entering ? 1.0f / mat.ior : mat.ior;
        float cosI = fabs(dot(rd, nFacing));
        float f0 = (1.0f - mat.ior) / (1.0f + mat.ior);
        f0 = f0 * f0;
        float fresnel = f0 + (1.0f - f0) * pow(1.0f - cosI, 5.0f);
        float3 refr = refract(rd, nFacing, eta);
        if (length_squared(refr) < 1e-8f || mini_rand(rng) < fresnel) {
            rd = reflect(rd, nFacing);
            ro = p + nFacing * 1e-4f;
        } else {
            throughput *= float3(mat.rgb);
            rd = normalize(refr);
            ro = p - nFacing * 1e-4f;
        }
    }
}

// ==========================================================================
// Megakernel mode (M1)
// ==========================================================================

kernel void pathtraceKernel(constant MiniParams& P                [[buffer(0)]],
                            device float4* accum                  [[buffer(1)]],
                            device const MiniMaterial* materials  [[buffer(2)]],
                            device const MiniObject* objects      [[buffer(3)]],
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
        if (!closestHit(ro, rd, objects, P.numObjects, hit))
            break;  // black environment, like the CUDA renderer without a skybox
        device const MiniObject& obj = objects[hit.objIdx];
        device const MiniMaterial& mat = materials[obj.materialId];
        if (mat.type == MINI_MAT_EMITTING) {
            L += throughput * float3(mat.rgb) * mat.emittance;
            break;
        }
        scatterMaterial(mat, ro + rd * hit.t, hit.n, ro, rd, throughput, rng);
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
        atomic_store_explicit(&counts[WF_COUNT_RAY_A], P.width * P.height, memory_order_relaxed);
        atomic_store_explicit(&counts[WF_COUNT_RAY_B], 0u, memory_order_relaxed);
        atomic_store_explicit(&counts[WF_COUNT_SHADE], 0u, memory_order_relaxed);
    }
}

// Single-thread dispatch: turn a queue count into indirect threadgroup args
// (three uints per 16-byte slot) and reset the counter the next stage pushes
// into. This keeps the whole bounce loop free of CPU readbacks.
kernel void wf_prepare(constant WfCtl& C          [[buffer(0)]],
                       device atomic_uint* counts [[buffer(1)]],
                       device uint* args          [[buffer(2)]])
{
    uint c = atomic_load_explicit(&counts[C.srcCounter], memory_order_relaxed);
    args[C.argSlot * 4 + 0] = (c + PRIM_TILE - 1u) / PRIM_TILE;
    args[C.argSlot * 4 + 1] = 1u;
    args[C.argSlot * 4 + 2] = 1u;
    atomic_store_explicit(&counts[C.zeroCounter], 0u, memory_order_relaxed);
}

kernel void wf_intersect(constant WfCtl& C               [[buffer(0)]],
                         device atomic_uint* counts      [[buffer(1)]],
                         device const WfPath* raysIn     [[buffer(2)]],
                         device WfPath* shadeQueue       [[buffer(3)]],
                         device const MiniObject* objects [[buffer(4)]],
                         uint tid [[thread_position_in_grid]])
{
    if (tid >= atomic_load_explicit(&counts[C.srcCounter], memory_order_relaxed))
        return;
    WfPath path = raysIn[tid];
    MiniHit hit;
    if (!closestHit(float3(path.origin), float3(path.dir), objects, C.numObjects, hit))
        return;  // escaped: contributes nothing, simply not re-enqueued
    path.t = hit.t;
    path.normal = hit.n;
    path.matId = (uint)objects[hit.objIdx].materialId;
    shadeQueue[prim_queue_alloc(&counts[C.dstCounter])] = path;
}

kernel void wf_shade(constant WfCtl& C                   [[buffer(0)]],
                     device atomic_uint* counts          [[buffer(1)]],
                     device const WfPath* shadeQueue     [[buffer(2)]],
                     device WfPath* raysOut              [[buffer(3)]],
                     device const MiniMaterial* materials [[buffer(4)]],
                     device float4* accum                [[buffer(5)]],
                     uint tid [[thread_position_in_grid]])
{
    if (tid >= atomic_load_explicit(&counts[C.srcCounter], memory_order_relaxed))
        return;
    WfPath path = shadeQueue[tid];
    device const MiniMaterial& mat = materials[path.matId];

    if (mat.type == MINI_MAT_EMITTING) {
        // One path per pixel per sample, so this write does not race.
        float3 L = float3(path.throughput) * float3(mat.rgb) * mat.emittance;
        if (all(isfinite(L)))
            accum[path.pixel] += float4(L, 1.0f);
        return;
    }
    if (path.depth + 1 >= C.maxDepth)
        return;

    MiniRng rng;
    rng.state = path.rng;
    float3 ro = float3(path.origin);
    float3 rd = float3(path.dir);
    float3 throughput = float3(path.throughput);
    scatterMaterial(mat, ro + rd * path.t, float3(path.normal), ro, rd, throughput, rng);

    path.origin = ro;
    path.dir = rd;
    path.throughput = throughput;
    path.rng = rng.state;
    path.depth++;
    raysOut[prim_queue_alloc(&counts[C.dstCounter])] = path;
}
