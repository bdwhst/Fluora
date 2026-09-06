#ifndef CORE_MEDIUM_SHARED_H
#define CORE_MEDIUM_SHARED_H
// Participating media for the portable renderer core (the CUDA renderer's
// media.h / medium.h port, M4 part 2 step 4): the device-side medium record,
// the Henyey-Greenstein phase function, bricked density grids (the NanoVDB
// port) and the majorant-segment iterator the integrator's delta tracker
// walks. Single-source across MSL, CUDA and host C++ via gpu_portable.h
// (docs/portable-device-code.md). Under MSL this file is concatenated after
// spectrum_shared.h and bsdf_shared.h (bsdf_onb); elsewhere the #includes
// resolve.
//
// Grid media do not run NanoVDB on the device. The vendored NanoVDB (32.3)
// has CUDA/OpenCL/host personalities but none for MSL, and the runtime MSL
// build concatenates sources without resolving #includes, so its 7k-line
// header is out of reach there. Instead core/volume_loader.cpp reads the
// grid on the host (NanoVDB's own reader) and re-bricks it: 8^3 voxel bricks
// (NanoVDB's leaf size, so leaves copy straight across) addressed through a
// dense brick table over the grid's index-space bounding box, plus a per-
// brick majorant. Two flat buffers, offsets in the medium record — invariant
// I-1, no pointers — and the same code on every backend.

#ifndef __METAL_VERSION__
#include "../rhi/gpu_portable.h"
#include "spectrum_shared.h"
#include "bsdf_shared.h"
#endif

#define MEDIUM_HOMOGENEOUS 0
#define MEDIUM_GRID        1   // bricked density (+ optional temperature) grid

#define VOL_BRICK_DIM   8
#define VOL_BRICK_SIZE  512u
#define VOL_BRICK_EMPTY 0xFFFFFFFFu   // brick-table entry: all-zero brick
#define VOL_TABLE_NONE  0xFFFFFFFFu   // MediumGpu offset: grid absent
#define VOL_EMISSION_MIN_KELVIN 100.0f

// One "Media" entry. sigma_a/sigma_s are dense-spectrum offsets into the spd
// table (SIGMA_SCALE already applied on the host, RGB widened as an unbounded
// rgb2spec spectrum). Grid media add the index-space mapping and the offsets
// of their bricks: `*Table` into the volume table buffer (uint per brick),
// `*Voxels` and `majorants` into the volume data buffer (float). 240 bytes,
// host/device layout identical (the 4x4s come first among the 16-byte
// members, scalars in groups of four).
struct MediumGpu {
    unsigned int sigmaASpd;
    unsigned int sigmaSSpd;
    float g;                   // Henyey-Greenstein asymmetry, (-1, 1)
    unsigned int type;         // MEDIUM_*
    gpu_storage4x4 indexFromWorld;       // grid: world -> density voxel index space
    // The temperature grid carries its own map (a VDB file's density and
    // temperature grids need not share voxel size or origin — ground_explosion
    // has 0.15 vs 0.25 voxels plus a translation; NanoVDBMedium::Le in the
    // CUDA renderer likewise queries worldToIndexF per grid).
    gpu_storage4x4 tempIndexFromWorld;   // grid: world -> temperature voxel index space
    int gridMinX, gridMinY, gridMinZ;   // index-space origin of brick (0,0,0), multiple of 8
    int pad0;
    unsigned int brickDimX, brickDimY, brickDimZ;   // bricks per axis
    unsigned int pad1;
    unsigned int densityTable;   // grid: brick table offset (uints)
    unsigned int densityVoxels;  // grid: voxel data offset (floats)
    unsigned int majorants;      // grid: per-brick max density offset (floats)
    unsigned int tempTable;      // grid emission: temperature brick table, or VOL_TABLE_NONE
    unsigned int tempVoxels;
    float leScale;               // LESCALE (0 = no emission)
    float tempScale;             // TEMPSCALE: kelvin = (grid - TEMPOFFSET) * TEMPSCALE
    float tempOffset;
    int tempMinX, tempMinY, tempMinZ;   // the temperature grid's own brick extents,
    int pad2;                           // in its own index space (tempIndexFromWorld)
    unsigned int tempDimX, tempDimY, tempDimZ;
    unsigned int pad3;
};

// Beer-Lambert transmittance over distance t. Written per component so an
// infinite t (the ray escaped while inside the medium) gives exactly 0 for
// sigma_t > 0 and exactly 1 for sigma_t == 0, never 0*inf.
GPU_FN inline GpuSpectrum medium_transmittance(GpuSpectrum sigmaT, float t)
{
    GpuSpectrum T;
    for (int i = 0; i < SPD_N_SAMPLES; i++)
        T[i] = sigmaT[i] > 0.0f ? exp(-sigmaT[i] * t) : 1.0f;
    return T;
}

// Henyey-Greenstein phase function, PBRT convention: wo points back toward
// the previous vertex, so cosTheta = dot(wo, wi) and g > 0 peaks at
// wi = -wo (forward scattering). Normalized over the sphere, so it is its own
// sampling pdf below.
GPU_FN inline float hg_phase(float cosTheta, float g)
{
    float denom = 1.0f + g * g + 2.0f * g * cosTheta;
    return (1.0f / (4.0f * GPU_PI)) * (1.0f - g * g) / (denom * sqrt(max(denom, 0.0f)));
}

// Samples wi with pdf = hg_phase(dot(wo, wi), g) (SampleHenyeyGreenstein;
// |g| < 1e-3 falls back to the uniform sphere). Returns the pdf/phase value.
GPU_FN inline float hg_sample(gpu_float3 wo, float g, float u1, float u2,
                              GPU_THREAD gpu_float3& wi)
{
    float cosTheta;
    if (fabs(g) < 1e-3f) {
        cosTheta = 1.0f - 2.0f * u1;
    } else {
        float sq = (1.0f - g * g) / (1.0f + g - 2.0f * g * u1);
        cosTheta = -1.0f / (2.0f * g) * (1.0f + g * g - sq * sq);
    }
    float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * GPU_PI * u2;
    gpu_float3 b1, b2;
    bsdf_onb(wo, b1, b2);
    wi = normalize(b1 * (sinTheta * cos(phi)) + b2 * (sinTheta * sin(phi)) + wo * cosTheta);
    return hg_phase(cosTheta, g);
}

// ---------------------------------------------------------------------------
// Bricked grids
// ---------------------------------------------------------------------------

// One grid's addressing, resolved from a MediumGpu (density or temperature).
struct VolGrid {
    GPU_DEVICE const unsigned int* table;
    GPU_DEVICE const float* data;
    unsigned int tableOff;
    unsigned int voxelOff;
    int minX, minY, minZ;
    unsigned int dimX, dimY, dimZ;
};

GPU_FN inline VolGrid vol_density_grid(GPU_DEVICE const unsigned int* table,
                                       GPU_DEVICE const float* data, GPU_DEVICE const MediumGpu& m)
{
    VolGrid g;
    g.table = table;
    g.data = data;
    g.tableOff = m.densityTable;
    g.voxelOff = m.densityVoxels;
    g.minX = m.gridMinX;
    g.minY = m.gridMinY;
    g.minZ = m.gridMinZ;
    g.dimX = m.brickDimX;
    g.dimY = m.brickDimY;
    g.dimZ = m.brickDimZ;
    return g;
}

GPU_FN inline VolGrid vol_temperature_grid(GPU_DEVICE const unsigned int* table,
                                           GPU_DEVICE const float* data,
                                           GPU_DEVICE const MediumGpu& m)
{
    VolGrid g;
    g.table = table;
    g.data = data;
    g.tableOff = m.tempTable;
    g.voxelOff = m.tempVoxels;
    g.minX = m.tempMinX;
    g.minY = m.tempMinY;
    g.minZ = m.tempMinZ;
    g.dimX = m.tempDimX;
    g.dimY = m.tempDimY;
    g.dimZ = m.tempDimZ;
    return g;
}

// Voxel value at absolute index coordinates; 0 outside the bricks or in an
// empty brick. Voxels within a brick are laid out like NanoVDB leaves
// (x-major, z fastest), which is how the loader copies them.
GPU_FN inline float vol_voxel(GPU_THREAD const VolGrid& g, int ix, int iy, int iz)
{
    int rx = ix - g.minX, ry = iy - g.minY, rz = iz - g.minZ;
    if (rx < 0 || ry < 0 || rz < 0 || rx >= (int)(g.dimX * VOL_BRICK_DIM)
        || ry >= (int)(g.dimY * VOL_BRICK_DIM) || rz >= (int)(g.dimZ * VOL_BRICK_DIM))
        return 0.0f;
    unsigned int b = ((unsigned int)(rz >> 3) * g.dimY + (unsigned int)(ry >> 3)) * g.dimX
                   + (unsigned int)(rx >> 3);
    unsigned int e = g.table[g.tableOff + b];
    if (e == VOL_BRICK_EMPTY)
        return 0.0f;
    unsigned int o = (unsigned int)(((rx & 7) << 6) | ((ry & 7) << 3) | (rz & 7));
    return g.data[g.voxelOff + e * VOL_BRICK_SIZE + o];
}

// Trilinear sample at an index-space point; voxel values sit at integer
// coordinates (NanoVDB SampleFromVoxels, order 1).
GPU_FN inline float vol_sample(GPU_THREAD const VolGrid& g, gpu_float3 p)
{
    float fx = floor(p.x), fy = floor(p.y), fz = floor(p.z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    float tx = p.x - fx, ty = p.y - fy, tz = p.z - fz;
    float c00 = vol_voxel(g, ix, iy, iz) * (1.0f - tx) + vol_voxel(g, ix + 1, iy, iz) * tx;
    float c10 = vol_voxel(g, ix, iy + 1, iz) * (1.0f - tx) + vol_voxel(g, ix + 1, iy + 1, iz) * tx;
    float c01 = vol_voxel(g, ix, iy, iz + 1) * (1.0f - tx) + vol_voxel(g, ix + 1, iy, iz + 1) * tx;
    float c11 = vol_voxel(g, ix, iy + 1, iz + 1) * (1.0f - tx)
              + vol_voxel(g, ix + 1, iy + 1, iz + 1) * tx;
    float c0 = c00 * (1.0f - ty) + c10 * ty;
    float c1 = c01 * (1.0f - ty) + c11 * ty;
    return c0 * (1.0f - tz) + c1 * tz;
}

// ---------------------------------------------------------------------------
// Majorant segments (PBRT RayMajorantIterator): a homogeneous medium is one
// segment with sigma_maj = sigma_t; a grid medium is a 3D DDA over its bricks,
// each segment scaled by that brick's majorant density. t stays in world
// units throughout (the ray is transformed to index space with an unnormalized
// direction), so sigma per world unit applies directly.
// ---------------------------------------------------------------------------

struct VolMajorantIter {
    bool grid;
    bool done;              // homogeneous: single segment consumed
    float tMin, tMax;       // remaining span
    // DDA state (grid)
    float nextT[3];
    float deltaT[3];
    int voxel[3];
    int step[3];
    int limit[3];
    unsigned int dimX, dimY, dimZ;
};

// Prepares the iteration for the ray (ro, rd) over [0, tMax). For grids the
// ray is clipped to the brick bounds; outside them the medium contributes
// nothing (no segments).
GPU_FN inline void vol_majorant_init(GPU_THREAD VolMajorantIter& it, GPU_DEVICE const MediumGpu& m,
                                     gpu_float3 ro, gpu_float3 rd, float tMax)
{
    it.grid = m.type == MEDIUM_GRID;
    it.done = false;
    it.tMin = 0.0f;
    it.tMax = tMax;
    if (!it.grid)
        return;
    it.dimX = m.brickDimX;
    it.dimY = m.brickDimY;
    it.dimZ = m.brickDimZ;
    // Ray in brick units: index space minus the brick origin, over 8.
    gpu_float4x4 ifw = gpu_load4x4(m.indexFromWorld);
    gpu_float3 oI = gpu_xyz(ifw * gpu_float4(ro, 1.0f));
    gpu_float3 dI = gpu_xyz(ifw * gpu_float4(rd, 0.0f));
    gpu_float3 o = (oI - gpu_float3((float)m.gridMinX, (float)m.gridMinY, (float)m.gridMinZ))
                 * (1.0f / (float)VOL_BRICK_DIM);
    gpu_float3 d = dI * (1.0f / (float)VOL_BRICK_DIM);
    gpu_float3 dims = gpu_float3((float)it.dimX, (float)it.dimY, (float)it.dimZ);
    // Slab clip against [0, dims).
    float tEnter = 0.0f, tExit = tMax;
    float oo[3] = { o.x, o.y, o.z };
    float dd[3] = { d.x, d.y, d.z };
    float ext[3] = { dims.x, dims.y, dims.z };
    for (int a = 0; a < 3; a++) {
        if (dd[a] == 0.0f) {
            if (oo[a] < 0.0f || oo[a] >= ext[a])
                tEnter = INFINITY;
            continue;
        }
        float inv = 1.0f / dd[a];
        float t0 = (0.0f - oo[a]) * inv;
        float t1 = (ext[a] - oo[a]) * inv;
        tEnter = max(tEnter, min(t0, t1));
        tExit = min(tExit, max(t0, t1));
    }
    if (!(tEnter < tExit)) {
        it.done = true;
        return;
    }
    it.tMin = tEnter;
    it.tMax = tExit;
    float gi[3] = { oo[0] + dd[0] * tEnter, oo[1] + dd[1] * tEnter, oo[2] + dd[2] * tEnter };
    int res[3] = { (int)it.dimX, (int)it.dimY, (int)it.dimZ };
    for (int a = 0; a < 3; a++) {
        int v = (int)gi[a];
        it.voxel[a] = v < 0 ? 0 : (v >= res[a] ? res[a] - 1 : v);
        float dir = dd[a] == 0.0f ? 0.0f : dd[a];   // fold -0 into +0
        it.deltaT[a] = dir == 0.0f ? INFINITY : 1.0f / fabs(dir);
        if (dir >= 0.0f) {
            float nextPos = (float)(it.voxel[a] + 1);
            it.nextT[a] = dir == 0.0f ? INFINITY : tEnter + (nextPos - gi[a]) / dir;
            it.step[a] = 1;
            it.limit[a] = res[a];
        } else {
            float nextPos = (float)it.voxel[a];
            it.nextT[a] = tEnter + (nextPos - gi[a]) / dir;
            it.step[a] = -1;
            it.limit[a] = -1;
        }
    }
}

// Next segment: [segMin, segMax) and the majorant density factor (1 for a
// homogeneous medium, the brick's max density for a grid). False when the
// span is exhausted.
GPU_FN inline bool vol_majorant_next(GPU_THREAD VolMajorantIter& it, GPU_DEVICE const MediumGpu& m,
                                     GPU_DEVICE const float* volData,
                                     GPU_THREAD float& segMin, GPU_THREAD float& segMax,
                                     GPU_THREAD float& majorant)
{
    if (it.done)
        return false;
    if (!it.grid) {
        it.done = true;
        segMin = it.tMin;
        segMax = it.tMax;
        majorant = 1.0f;
        return true;
    }
    if (it.tMin >= it.tMax) {
        it.done = true;
        return false;
    }
    int bits = ((it.nextT[0] < it.nextT[1]) ? 4 : 0) + ((it.nextT[0] < it.nextT[2]) ? 2 : 0)
             + ((it.nextT[1] < it.nextT[2]) ? 1 : 0);
    const int cmpToAxis[8] = { 2, 1, 2, 1, 2, 2, 0, 0 };
    int axis = cmpToAxis[bits];
    float tVoxelExit = min(it.tMax, it.nextT[axis]);
    unsigned int brick = ((unsigned int)it.voxel[2] * it.dimY + (unsigned int)it.voxel[1]) * it.dimX
                       + (unsigned int)it.voxel[0];
    segMin = it.tMin;
    segMax = tVoxelExit;
    majorant = volData[m.majorants + brick];
    it.tMin = tVoxelExit;
    if (it.nextT[axis] > it.tMax)
        it.tMin = it.tMax;
    it.voxel[axis] += it.step[axis];
    if (it.voxel[axis] == it.limit[axis])
        it.tMin = it.tMax;
    it.nextT[axis] += it.deltaT[axis];
    return true;
}

#endif // CORE_MEDIUM_SHARED_H
