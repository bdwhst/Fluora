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

#define MINI_MAT_DIFFUSE  0
#define MINI_MAT_EMITTING 1
#define MINI_MAT_GLASS    2
#define MINI_MAT_MIRROR   3

struct MiniMaterial {
    mini_float3 rgb;
    int   type;         // MINI_MAT_*
    float emittance;
    float ior;
    float roughness;    // parsed but unused in M1 (microfacet renders as mirror)
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
    unsigned int pad0;
};

#endif // MINI_SHARED_H
