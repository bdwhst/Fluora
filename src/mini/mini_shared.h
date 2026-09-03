#ifndef MINI_SHARED_H
#define MINI_SHARED_H
// Structs shared between the FluoraMini host code and its MSL kernels
// (portability invariant I-3 in docs/metal-rhi-design.md). Compiled by both
// host clang (simd types) and the Metal compiler (native types) — the pairs
// below have identical size/alignment. Keep fields 16-byte-friendly: float3
// members first, scalars packed in groups of four.
//
// NOTE: under MSL this header is concatenated after gpu_portable.h (runtime
// MSL compilation cannot resolve #includes, so the #include below is inert
// there and the concat order must put gpu_portable.h first).

#ifndef __METAL_VERSION__
#include "../rhi/gpu_portable.h"
#endif
typedef gpu_storage3   mini_float3;
typedef gpu_storage4x4 mini_float4x4;

#define MINI_GEOM_CUBE   0
#define MINI_GEOM_SPHERE 1

#define MINI_MAT_DIFFUSE   0
#define MINI_MAT_EMITTING  1
#define MINI_MAT_GLASS     2
#define MINI_MAT_CONDUCTOR 3  // GGX; roughness < 1e-3 degenerates to mirror

#define MINI_ENV_NONE 0xFFFFFFFFu  // envMapIdx sentinel: black environment
#define MINI_TEX_NONE 0xFFFFFFFFu  // texIdx sentinel: untextured

// CUDA parity note (scene.cpp): env-map radiance is clamped to
// environmentMapMaxLumin, default vec3(1e5) for .txt scenes.
#define MINI_ENV_MAX_RADIANCE 1e5f

struct MiniMaterial {
    mini_float3 rgb;
    int   type;         // MINI_MAT_*
    float emittance;
    float ior;          // dielectric eta when etaSpd == SPD_NONE
    float roughness;    // conductor GGX alpha (CUDA uses it unsquared)
    unsigned int texIdx;  // base-color texture-heap index, or MINI_TEX_NONE
    // Dense-spectrum offsets into the spd table buffer, or SPD_NONE:
    // dielectric etaSpd -> dispersive eta (terminates secondary wavelengths);
    // conductor etaSpd/kSpd -> measured complex IOR, else PBRT reflectance
    // mode derives k from rgb (eta = 1).
    unsigned int etaSpd;
    unsigned int kSpd;
    unsigned int pad0;
};

// Unit primitives (cube [-0.5,0.5]^3, sphere r=0.5) with baked transforms,
// matching the original CIS-5650 convention so scene SCALE values line up.
struct MiniObject {
    mini_float4x4 transform;
    mini_float4x4 invTransform;
    mini_float4x4 invTranspose;
    int geomType;       // MINI_GEOM_*
    int materialId;
    int pad0, pad1;
};

struct MiniParams {
    mini_float3 camPos;
    mini_float3 camView;
    mini_float3 camUp;
    mini_float3 camRight;
    // Film output matrix rows (sRGB RGBFromXYZ, host-derived like
    // RGBColorSpace): outputRGB = (dot(filmR0,xyz), dot(filmR1,xyz), ...).
    mini_float3 filmR0;
    mini_float3 filmR1;
    mini_float3 filmR2;
    float pixelLenX;
    float pixelLenY;
    unsigned int width;
    unsigned int height;
    unsigned int iter;        // current sample index, mixed into the RNG seed
    unsigned int maxDepth;
    unsigned int numObjects;
    unsigned int bvhNumNodes; // RayIntersector::numNodes() (0 = no meshes)
    unsigned int envMapIdx;   // texture-heap index of the env map, or MINI_ENV_NONE
    unsigned int numLights;   // RtLight records (0 = no next-event estimation)
    unsigned int envW;        // env-map / distribution size (light_shared.h ENVDIST_*)
    unsigned int envH;
};

// ---- wavefront mode (design doc M2/M3) ----
// Path queues ping-pong between two ray buffers. Shading is queue-routed per
// material type (tier-1 of the get_bxdf plan): intersect looks at the hit
// material and pushes into that type's queue; each shade kernel runs one BSDF
// with zero divergence. Emissive hits are resolved inline in intersect.
#define WF_COUNT_RAY_A          0
#define WF_COUNT_RAY_B          1
#define WF_COUNT_SHADE_DIFFUSE  2
#define WF_COUNT_SHADE_CONDUCTOR 3
#define WF_COUNT_SHADE_GLASS    4
#define WF_COUNT_SHADOW         5  // shadow rays pushed by the shade kernels
#define WF_NUM_COUNTERS 8  // padded

// Indirect-args slots (16-byte stride each)
#define WF_ARG_INTERSECT 0
#define WF_ARG_DIFFUSE   1
#define WF_ARG_CONDUCTOR 2
#define WF_ARG_GLASS     3
#define WF_ARG_SHADOW    4
#define WF_NUM_ARG_SLOTS 5

// WfPath is device-only (the host never reads paths); the host allocates
// queues with this stride and the device static_asserts the real sizeof
// matches. Spectral state per path: throughput is one float4 spectrum, and
// wavelengths are recomputed each stage from lambdaU (+ the dispersion flag)
// instead of being carried — 8 bytes of queue traffic instead of 32.
#define WF_PATHSTATE_SIZE 96
#define WF_FLAG_SECONDARY_TERMINATED 1u
#define WF_FLAG_PREV_SPECULAR        2u  // last scatter was a delta BSDF (no MIS on hit)

// Shadow rays (next-event estimation) carry the already MIS-weighted film RGB
// contribution; wf_shadow adds it when the light is visible.
#define WF_SHADOWRAY_SIZE 48

// Per-dispatch control block for the wavefront kernels.
struct WfCtl {
    unsigned int srcCounter;   // queue this dispatch consumes
    unsigned int dstCounter;   // queue this dispatch pushes to
    unsigned int zeroCounter;  // wf_prepare: counter to reset
    unsigned int numLights;
    unsigned int numObjects;
    unsigned int maxDepth;
    unsigned int bvhNumNodes;
    unsigned int envMapIdx;
    unsigned int envW;
    unsigned int envH;
    unsigned int pad0, pad1;
    mini_float3 filmR0;        // film matrix rows, as in MiniParams
    mini_float3 filmR1;
    mini_float3 filmR2;
};

#endif // MINI_SHARED_H
