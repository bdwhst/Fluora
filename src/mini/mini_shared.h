#ifndef MINI_SHARED_H
#define MINI_SHARED_H
// Structs shared between the FluoraMini host code and its MSL kernels
// (portability invariant I-3 in docs/metal-rhi-design.md). Compiled by both
// host clang (simd types) and the Metal compiler (native types) — the pairs
// below have identical size/alignment. Keep fields 16-byte-friendly: float3
// members first, scalars packed in groups of four.
//
// NOTE: at runtime this header is textually prepended to pathtrace.metal
// before newLibraryWithSource (runtime MSL compilation cannot resolve
// #includes) — keep it self-contained.

#ifdef __METAL_VERSION__
#include <metal_stdlib>
typedef metal::float3   mini_float3;
typedef metal::float4x4 mini_float4x4;
#else
#include <simd/simd.h>
typedef simd_float3   mini_float3;
typedef simd_float4x4 mini_float4x4;
#endif

#define MINI_GEOM_CUBE   0
#define MINI_GEOM_SPHERE 1

#define MINI_MAT_DIFFUSE   0
#define MINI_MAT_EMITTING  1
#define MINI_MAT_GLASS     2
#define MINI_MAT_CONDUCTOR 3  // GGX; roughness < 1e-3 degenerates to mirror

#define MINI_ENV_NONE 0xFFFFFFFFu  // envMapIdx sentinel: black environment
#define MINI_TEX_NONE 0xFFFFFFFFu  // texIdx sentinel: untextured

struct MiniMaterial {
    mini_float3 rgb;
    int   type;         // MINI_MAT_*
    float emittance;
    float ior;
    float roughness;
    unsigned int texIdx;  // base-color texture-heap index, or MINI_TEX_NONE
    unsigned int pad0, pad1, pad2;
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
    float pixelLenX;
    float pixelLenY;
    unsigned int width;
    unsigned int height;
    unsigned int iter;        // current sample index, mixed into the RNG seed
    unsigned int maxDepth;
    unsigned int numObjects;
    unsigned int bvhNumNodes; // RayIntersector::numNodes() (0 = no meshes)
    unsigned int envMapIdx;   // texture-heap index of the env map, or MINI_ENV_NONE
    unsigned int pad0, pad1, pad2;
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
#define WF_NUM_COUNTERS 8  // padded

// Indirect-args slots (16-byte stride each)
#define WF_ARG_INTERSECT 0
#define WF_ARG_DIFFUSE   1
#define WF_ARG_CONDUCTOR 2
#define WF_ARG_GLASS     3
#define WF_NUM_ARG_SLOTS 4

// WfPath is MSL-only (the host never reads paths); the host allocates queues
// with this stride and MSL static_asserts the real sizeof matches.
#define WF_PATHSTATE_SIZE 80

// Per-dispatch control block for the wavefront kernels.
struct WfCtl {
    unsigned int srcCounter;   // queue this dispatch consumes
    unsigned int dstCounter;   // queue this dispatch pushes to
    unsigned int zeroCounter;  // wf_prepare: counter to reset
    unsigned int argSlot;      // wf_prepare: indirect-args slot (16-byte stride)
    unsigned int numObjects;
    unsigned int maxDepth;
    unsigned int bvhNumNodes;
    unsigned int envMapIdx;
};

#endif // MINI_SHARED_H
