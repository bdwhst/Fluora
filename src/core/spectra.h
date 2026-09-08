#pragma once
// Host-side spectral tables for the renderer core: builds the flat
// dense-spectra buffer consumed by core/spectrum_shared.h (invariant I-1 —
// device sees offsets, never pointers), ports of the CUDA renderer's
// spec::init / PiecewiseLinearSpectrum::from_interleaved / RGBColorSpace
// machinery. Data comes from SpectrumConsts/spectrum_tables.inl.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>

class SpectralTables {
public:
    // Preloads CIE X/Y/Z and the normalized D65 illuminant at the fixed
    // SPD_OFF_* offsets spectrum_shared.h assumes.
    SpectralTables();

    // Float offset of a named spectrum's dense 471-float run (the
    // scene-format names: glass-*, metal-*-eta / metal-*-k, stdillum-D65),
    // densified and appended on first request. SPD_NONE if unknown.
    uint32_t namedOffset(const std::string& name);

    // --- Editable slot pool ----------------------------------------------
    // Every device field that can hold an spd offset owns one slot for the
    // life of the scene: MiniMaterial::etaSpd/kSpd (mini_shared.h) and
    // MediumGpu::sigmaASpd/sigmaSSpd (medium_shared.h). Since each field holds
    // exactly one offset at a time, `2 * materials + 2 * media` is an exact
    // upper bound rather than a guess, so reserving it once means a slot
    // request can never fail and the buffer is never resized afterwards.
    //
    // That buys stable offsets: a live GUI edit rewrites its slot's
    // SPD_TABLE_SIZE floats in place, so the device record keeps its offset
    // and only that one run has to be re-uploaded. Slots are deliberately not
    // deduplicated — the pool is already sized for the worst case, so sharing
    // would free no memory while making an edit to one owner visible to
    // another.
    //
    // Call once, before writing any slot; the pool is appended after whatever
    // the table already holds (later namedOffset() calls append past it and
    // leave slot offsets untouched).
    void reservePool(uint32_t numSlots);
    uint32_t poolBaseOffset() const { return poolBase; }
    uint32_t poolSlotOffset(uint32_t slot) const;  // float offset into buffer()
    float* poolSlotData(uint32_t slot);            // SPD_TABLE_SIZE writable floats

    const std::vector<float>& buffer() const { return spd; }

private:
    std::vector<float> spd;
    std::unordered_map<std::string, uint32_t> cache;
    uint32_t poolBase = 0;
    uint32_t poolSlots = 0;
    bool poolReserved = false;   // distinguishes "not reserved" from a 0-slot pool
};

// Fill one dense SPD_TABLE_SIZE-float run in place. These are the two spectrum
// kinds the scene formats produce; they write through a destination pointer so
// the same code serves both the append-and-cache path above and live edits
// into a pool slot.
//
// spdWriteNamed returns false for an unknown name, leaving `dst` untouched.
// spdWriteRgbUnbounded widens an RGB coefficient as a PBRT
// RGBUnboundedSpectrum (scale * sigmoid polynomial over the sRGB table) — the
// form medium sigma_a / sigma_s need, since those exceed 1.
bool spdWriteNamed(const std::string& name, float* dst);
void spdWriteRgbUnbounded(const glm::vec3& rgb, float* dst);

// sRGB RGB->sigmoid-coefficient table (PBRT), for the rgb2spec device buffer:
// upload zNodes then coeffs back to back.
struct Rgb2SpecView {
    const float* zNodes;
    size_t zNodeCount;       // 64
    const float* coeffs;
    size_t coeffCount;       // 3*64*64*64*3
};
Rgb2SpecView rgb2specSrgb();

// The film's output matrix: sRGB RGBFromXYZ derived exactly as the
// RGBColorSpace constructor does (primaries + D65 white point).
glm::mat3 srgbRgbFromXyz();
