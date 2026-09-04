#include "volume_loader.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

// The scene .nvdb files are ZIP-compressed (io::Codec::ZIP), so the reader
// needs zlib on every platform (CMake links it for FluoraMini).
#define NANOVDB_USE_ZIP 1
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/HostBuffer.h>
#include <nanovdb/util/IO.h>

#include "medium_shared.h"

namespace {

// The grid's affine index map, recovered numerically from worldToIndexF so
// no assumption about NanoVDB's matrix storage convention is baked in.
glm::mat4 indexFromWorldOf(const nanovdb::FloatGrid& grid)
{
    auto w2i = [&](float x, float y, float z) {
        nanovdb::Vec3f p = grid.worldToIndexF(nanovdb::Vec3f(x, y, z));
        return glm::vec3(p[0], p[1], p[2]);
    };
    glm::vec3 o = w2i(0.0f, 0.0f, 0.0f);
    glm::vec3 ex = w2i(1.0f, 0.0f, 0.0f) - o;
    glm::vec3 ey = w2i(0.0f, 1.0f, 0.0f) - o;
    glm::vec3 ez = w2i(0.0f, 0.0f, 1.0f) - o;
    glm::mat4 m(1.0f);
    m[0] = glm::vec4(ex, 0.0f);
    m[1] = glm::vec4(ey, 0.0f);
    m[2] = glm::vec4(ez, 0.0f);
    m[3] = glm::vec4(o, 1.0f);
    return m;
}

} // namespace

bool loadNanoVdbGrid(const std::string& path, const std::string& gridName, BrickGrid& out,
                     std::string& err)
{
    nanovdb::GridHandle<nanovdb::HostBuffer> handle;
    try {
        handle = nanovdb::io::readGrid<nanovdb::HostBuffer>(path, gridName);
    } catch (const std::exception& e) {
        err = path + ": " + e.what();
        return false;
    }
    if (!handle) {
        err = path + ": no grid named '" + gridName + "'";
        return false;
    }
    const nanovdb::FloatGrid* grid = handle.grid<float>();
    if (!grid) {
        err = path + ": grid '" + gridName + "' is not a float grid";
        return false;
    }
    using LeafT = nanovdb::FloatGrid::TreeType::LeafNodeType;
    static_assert(LeafT::DIM == VOL_BRICK_DIM && LeafT::SIZE == VOL_BRICK_SIZE,
                  "brick layout mirrors NanoVDB leaves");

    const nanovdb::CoordBBox bbox = grid->indexBBox();   // inclusive
    if (bbox.empty()) {
        err = path + ": grid '" + gridName + "' is empty";
        return false;
    }
    auto floorDiv8 = [](int v) { return (v >= 0 ? v : v - 7) / 8; };
    glm::ivec3 lo(floorDiv8(bbox.min()[0]) * 8, floorDiv8(bbox.min()[1]) * 8,
                  floorDiv8(bbox.min()[2]) * 8);
    glm::ivec3 hi(bbox.max()[0], bbox.max()[1], bbox.max()[2]);   // inclusive
    glm::uvec3 dim((hi.x - lo.x) / 8 + 1, (hi.y - lo.y) / 8 + 1, (hi.z - lo.z) / 8 + 1);

    out = BrickGrid{};
    out.origin = lo;
    out.brickDim = dim;
    out.indexFromWorld = indexFromWorldOf(*grid);
    const size_t numBricks = (size_t)dim.x * dim.y * dim.z;
    out.table.assign(numBricks, VOL_BRICK_EMPTY);
    std::vector<float> brickMax(numBricks, 0.0f);

    auto acc = grid->getAccessor();
    std::vector<float> values(VOL_BRICK_SIZE);
    for (unsigned bz = 0; bz < dim.z; bz++) {
        for (unsigned by = 0; by < dim.y; by++) {
            for (unsigned bx = 0; bx < dim.x; bx++) {
                nanovdb::Coord ijk(lo.x + (int)bx * 8, lo.y + (int)by * 8, lo.z + (int)bz * 8);
                float vmax = 0.0f;
                if (const LeafT* leaf = acc.probeLeaf(ijk)) {
                    for (uint32_t i = 0; i < VOL_BRICK_SIZE; i++) {
                        float v = std::max(leaf->getValue(i), 0.0f);
                        values[i] = v;
                        vmax = std::max(vmax, v);
                    }
                } else {
                    // No leaf: the whole brick is one tile value (usually the
                    // background, 0; a filled interior tile is not).
                    float v = std::max(acc.getValue(ijk), 0.0f);
                    if (v <= 0.0f)
                        continue;
                    std::fill(values.begin(), values.end(), v);
                    vmax = v;
                }
                if (vmax <= 0.0f)
                    continue;
                size_t b = ((size_t)bz * dim.y + by) * dim.x + bx;
                out.table[b] = (uint32_t)(out.voxels.size() / VOL_BRICK_SIZE);
                out.voxels.insert(out.voxels.end(), values.begin(), values.end());
                brickMax[b] = vmax;
                out.maxValue = std::max(out.maxValue, vmax);
                for (float v : values)
                    out.activeVoxels += v > 0.0f ? 1 : 0;
            }
        }
    }

    // Majorants: a trilinear sample inside brick b reads voxels up to one
    // step outside it, so its bound is the max over b and its neighbours.
    out.majorants.assign(numBricks, 0.0f);
    for (unsigned bz = 0; bz < dim.z; bz++) {
        for (unsigned by = 0; by < dim.y; by++) {
            for (unsigned bx = 0; bx < dim.x; bx++) {
                float m = 0.0f;
                for (int dz = -1; dz <= 1; dz++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int nx = (int)bx + dx, ny = (int)by + dy, nz = (int)bz + dz;
                            if (nx < 0 || ny < 0 || nz < 0 || nx >= (int)dim.x || ny >= (int)dim.y
                                || nz >= (int)dim.z)
                                continue;
                            m = std::max(m, brickMax[((size_t)nz * dim.y + ny) * dim.x + nx]);
                        }
                    }
                }
                out.majorants[((size_t)bz * dim.y + by) * dim.x + bx] = m;
            }
        }
    }
    return true;
}
