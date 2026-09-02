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

    const std::vector<float>& buffer() const { return spd; }

private:
    std::vector<float> spd;
    std::unordered_map<std::string, uint32_t> cache;
};

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
