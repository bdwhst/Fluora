// FluoraMini megakernel path tracer (milestone M1, docs/metal-rhi-design.md).
// One dispatch = one sample per pixel, accumulated into `accum`. Brute-force
// intersection over analytic cubes/spheres — no BVH until M3. Camera and
// transform conventions replicate pathtrace.cu's generateRayFromCamera so the
// framing matches the CUDA renderer exactly.
//
// mini_shared.h is textually prepended before compilation; do not #include it.
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

    // Same jittered pinhole formula as generateRayFromCamera in pathtrace.cu.
    float jx = mini_rand(rng) - 0.5f;
    float jy = mini_rand(rng) - 0.5f;
    float3 rd = normalize(P.camView
        - P.camRight * P.pixelLenX * ((float)gid.x - (float)P.width * 0.5f + jx)
        - P.camUp    * P.pixelLenY * ((float)gid.y - (float)P.height * 0.5f + jy));
    float3 ro = P.camPos;

    float3 L = float3(0.0f);
    float3 throughput = float3(1.0f);

    for (uint depth = 0; depth < P.maxDepth; depth++) {
        MiniHit hit;
        if (!closestHit(ro, rd, objects, P.numObjects, hit))
            break;  // black environment, like the CUDA renderer without a skybox

        device const MiniObject& obj = objects[hit.objIdx];
        device const MiniMaterial& mat = materials[obj.materialId];
        float3 p = ro + rd * hit.t;
        float3 n = hit.n;
        float3 nFacing = dot(n, rd) < 0.0f ? n : -n;

        if (mat.type == MINI_MAT_EMITTING) {
            L += throughput * mat.rgb * mat.emittance;
            break;
        }
        else if (mat.type == MINI_MAT_DIFFUSE) {
            throughput *= mat.rgb;
            rd = cosineSampleHemisphere(nFacing, rng);
            ro = p + nFacing * 1e-4f;
        }
        else if (mat.type == MINI_MAT_MIRROR) {
            throughput *= mat.rgb;
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
                throughput *= mat.rgb;
                rd = normalize(refr);
                ro = p - nFacing * 1e-4f;
            }
        }
    }

    if (all(isfinite(L)))
        accum[idx] += float4(L, 1.0f);
}
