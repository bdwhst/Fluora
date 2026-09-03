#include "spectra.h"

#include <cmath>
#include <stdexcept>

#include "spectrum_shared.h"
#include "../../external/SpectrumConsts/spectrum_tables.inl"

namespace {

// Host ports of the CUDA renderer's spectrum machinery, same float
// expressions: find_interval bisection, piecewise lerp, from_interleaved's
// range padding, and CIE-Y-integral normalization applied to the piecewise
// values BEFORE densification (spectrum.cu order).

int findInterval(int sz, const std::vector<float>& lambdas, float lambda)
{
    int size = sz - 2, first = 1;
    while (size > 0) {
        int half = size >> 1, mid = first + half;
        bool pred = lambdas[mid] <= lambda;
        first = pred ? mid + 1 : first;
        size = pred ? size - half - 1 : half;
    }
    return std::min(std::max(first - 1, 0), sz - 2);
}

struct Piecewise {
    std::vector<float> lambdas, values;

    float operator()(float lambda) const
    {
        if (lambdas.empty() || lambda < lambdas.front() || lambda > lambdas.back())
            return 0.0f;
        int i = findInterval((int)lambdas.size(), lambdas, lambda);
        float a = (lambda - lambdas[i]) / (lambdas[i + 1] - lambdas[i]);
        return (1.0f - a) * values[i] + a * values[i + 1];
    }

    void scale(float s)
    {
        for (float& v : values)
            v *= s;
    }
};

Piecewise fromPairs(const float* lambdas, const float* values, size_t n)
{
    Piecewise p;
    p.lambdas.assign(lambdas, lambdas + n);
    p.values.assign(values, values + n);
    return p;
}

// Dense CIE Y curve, needed for illuminant normalization before the table
// exists; built once.
const Piecewise& cieY()
{
    static Piecewise y = fromPairs(spec::CIE_lambda, spec::CIE_Y, spec::nCIESamples);
    return y;
}

std::vector<float> densify(const Piecewise& p)
{
    std::vector<float> v(SPD_TABLE_SIZE);
    for (int l = (int)SPD_LAMBDA_MIN; l <= (int)SPD_LAMBDA_MAX; l++)
        v[l - (int)SPD_LAMBDA_MIN] = p((float)l);
    return v;
}

// inner_product(f, g): unit-spaced integral over [360, 830]. g is the dense
// CIE Y here (integer lambdas, so dense-vs-piecewise sampling agrees).
float innerProductWithY(const Piecewise& f)
{
    float integral = 0;
    for (float lambda = SPD_LAMBDA_MIN; lambda <= SPD_LAMBDA_MAX; ++lambda)
        integral += f(lambda) * cieY()(lambda);
    return integral;
}

// PiecewiseLinearSpectrum::from_interleaved: (lambda, value) pairs with range
// padding; normalize scales illuminants to the CIE Y integral.
Piecewise fromInterleaved(const float* samples, size_t pairCount, bool normalize)
{
    Piecewise p;
    if (samples[0] > SPD_LAMBDA_MIN) {
        p.lambdas.push_back(SPD_LAMBDA_MIN - 1);
        p.values.push_back(samples[1]);
    }
    for (size_t i = 0; i < pairCount; i++) {
        p.lambdas.push_back(samples[i << 1]);
        p.values.push_back(samples[(i << 1) + 1]);
    }
    if (p.lambdas.back() < SPD_LAMBDA_MAX) {
        p.lambdas.push_back(SPD_LAMBDA_MAX + 1);
        p.values.push_back(p.values.back());
    }
    if (normalize)
        p.scale(SPD_CIE_Y_INTEGRAL / innerProductWithY(p));
    return p;
}

#define INTERLEAVED(arr, normalize) \
    fromInterleaved((const float*)(arr), sizeof(arr) / sizeof(float) / 2, normalize)

// The named-spectrum registry subset the scene format reaches (spectrum_data
// keeps the full CUDA-side map; consolidate in M4).
Piecewise namedPiecewise(const std::string& name, bool& ok)
{
    using namespace spec;
    ok = true;
    if (name == "glass-BK7") return INTERLEAVED(GlassBK7_eta, false);
    if (name == "glass-BAF10") return INTERLEAVED(GlassBAF10_eta, false);
    if (name == "glass-FK51A") return INTERLEAVED(GlassFK51A_eta, false);
    if (name == "glass-LASF9") return INTERLEAVED(GlassLASF9_eta, false);
    if (name == "glass-F5") return INTERLEAVED(GlassSF5_eta, false);
    if (name == "glass-F10") return INTERLEAVED(GlassSF10_eta, false);
    if (name == "glass-F11") return INTERLEAVED(GlassSF11_eta, false);
    if (name == "glass-Fake") return INTERLEAVED(GlassSFake_eta, false);
    if (name == "metal-Ag-eta") return INTERLEAVED(Ag_eta, false);
    if (name == "metal-Ag-k") return INTERLEAVED(Ag_k, false);
    if (name == "metal-Al-eta") return INTERLEAVED(Al_eta, false);
    if (name == "metal-Al-k") return INTERLEAVED(Al_k, false);
    if (name == "metal-Au-eta") return INTERLEAVED(Au_eta, false);
    if (name == "metal-Au-k") return INTERLEAVED(Au_k, false);
    if (name == "metal-Cu-eta") return INTERLEAVED(Cu_eta, false);
    if (name == "metal-Cu-k") return INTERLEAVED(Cu_k, false);
    if (name == "metal-CuZn-eta") return INTERLEAVED(CuZn_eta, false);
    if (name == "metal-CuZn-k") return INTERLEAVED(CuZn_k, false);
    if (name == "metal-MgO-eta") return INTERLEAVED(MgO_eta, false);
    if (name == "metal-MgO-k") return INTERLEAVED(MgO_k, false);
    if (name == "metal-TiO2-eta") return INTERLEAVED(TiO2_eta, false);
    if (name == "metal-TiO2-k") return INTERLEAVED(TiO2_k, false);
    if (name == "stdillum-D65") return INTERLEAVED(CIE_Illum_D6500, true);
    ok = false;
    return Piecewise{};
}

} // namespace

SpectralTables::SpectralTables()
{
    auto append = [&](const std::vector<float>& dense) {
        uint32_t off = (uint32_t)spd.size();
        spd.insert(spd.end(), dense.begin(), dense.end());
        return off;
    };
    // Fixed offsets, in the order spectrum_shared.h's SPD_OFF_* constants
    // assume.
    append(densify(fromPairs(spec::CIE_lambda, spec::CIE_X, spec::nCIESamples)));
    append(densify(fromPairs(spec::CIE_lambda, spec::CIE_Y, spec::nCIESamples)));
    append(densify(fromPairs(spec::CIE_lambda, spec::CIE_Z, spec::nCIESamples)));
    uint32_t d65 = append(densify(INTERLEAVED(spec::CIE_Illum_D6500, true)));
    if (d65 != SPD_OFF_ILLUM_D65 || spd.size() != SPD_FIXED_TABLE_FLOATS)
        throw std::runtime_error("spectral table layout mismatch");
    cache.emplace("stdillum-D65", d65);
}

uint32_t SpectralTables::namedOffset(const std::string& name)
{
    auto it = cache.find(name);
    if (it != cache.end())
        return it->second;
    bool ok = false;
    Piecewise p = namedPiecewise(name, ok);
    if (!ok)
        return SPD_NONE;
    uint32_t off = (uint32_t)spd.size();
    std::vector<float> dense = densify(p);
    spd.insert(spd.end(), dense.begin(), dense.end());
    cache.emplace(name, off);
    return off;
}

extern const int sRGBToSpectrumTable_Res;
extern const float sRGBToSpectrumTable_Scale[64];
extern const float sRGBToSpectrumTable_Data[3][64][64][64][3];

Rgb2SpecView rgb2specSrgb()
{
    Rgb2SpecView v;
    v.zNodes = sRGBToSpectrumTable_Scale;
    v.zNodeCount = 64;
    v.coeffs = &sRGBToSpectrumTable_Data[0][0][0][0][0];
    v.coeffCount = (size_t)3 * 64 * 64 * 64 * 3;
    return v;
}

glm::mat3 srgbRgbFromXyz()
{
    // RGBColorSpace ctor for sRGB (color.cu): BT.709 primaries, D65 white.
    auto xyYtoXYZ = [](glm::vec2 xy, float Y) {
        if (xy.y == 0)
            return glm::vec3(0);
        return glm::vec3(xy.x * Y / xy.y, Y, (1 - xy.x - xy.y) * Y / xy.y);
    };
    Piecewise d65 = INTERLEAVED(spec::CIE_Illum_D6500, true);
    Piecewise cx = fromPairs(spec::CIE_lambda, spec::CIE_X, spec::nCIESamples);
    Piecewise cz = fromPairs(spec::CIE_lambda, spec::CIE_Z, spec::nCIESamples);
    glm::vec3 W(0.0f);
    for (float l = SPD_LAMBDA_MIN; l <= SPD_LAMBDA_MAX; ++l) {
        W.x += cx(l) * d65(l);
        W.y += cieY()(l) * d65(l);
        W.z += cz(l) * d65(l);
    }
    W /= SPD_CIE_Y_INTEGRAL;

    glm::vec3 R = xyYtoXYZ(glm::vec2(.64f, .33f), 1.0f);
    glm::vec3 G = xyYtoXYZ(glm::vec2(.3f, .6f), 1.0f);
    glm::vec3 B = xyYtoXYZ(glm::vec2(.15f, .06f), 1.0f);
    glm::mat3 rgb;
    rgb[0] = R;
    rgb[1] = G;
    rgb[2] = B;
    glm::vec3 C = glm::inverse(rgb) * W;
    glm::mat3 diagC(0.0f);
    diagC[0][0] = C.x;
    diagC[1][1] = C.y;
    diagC[2][2] = C.z;
    glm::mat3 XYZFromRGB = rgb * diagC;
    return glm::inverse(XYZFromRGB);
}
