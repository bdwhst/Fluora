#include "color.h"
#include "glm/gtx/matrix_operation.hpp"

#include "utilities.h"
RGBSigmoidPolynomial RGBToSpectrumTable::operator()(const glm::vec3& rgb) const
{
    // Handle uniform _rgb_ values
    if (rgb[0] == rgb[1] && rgb[1] == rgb[2])
        return RGBSigmoidPolynomial(0, 0,
            (rgb[0] - .5f) / std::sqrt(std::max(rgb[0] * (1 - rgb[0]),0.0f)));

    // Find maximum component and compute remapped component values
    int maxc =
        (rgb[0] > rgb[1]) ? ((rgb[0] > rgb[2]) ? 0 : 2) : ((rgb[1] > rgb[2]) ? 1 : 2);
    float z = rgb[maxc];
    float x = rgb[(maxc + 1) % 3] * (res - 1) / z;
    float y = rgb[(maxc + 2) % 3] * (res - 1) / z;

    // Compute integer indices and offsets for coefficient interpolation
    int xi = math::clamp((int)x, 0, res - 2), yi = math::clamp((int)y, 0, res - 2),
        zi = math::find_interval(res, [&](int i) { return zNodes[i] < z; });
    /*int xi = 0, yi = 0,
        zi = 0;*/
    float dx = x - xi, dy = y - yi, dz = (z - zNodes[zi]) / (zNodes[zi + 1] - zNodes[zi]);

    // Trilinearly interpolate sigmoid polynomial coefficients _c_
    float c[3];
    for (int i = 0; i < 3; ++i) {
        // Define _co_ lambda for looking up sigmoid polynomial coefficients
        auto co = [&](int dx, int dy, int dz) {
            return (*coeffs)[maxc][zi + dz][yi + dy][xi + dx][i];
            };

        c[i] = math::lerp(dz,
            math::lerp(dy, math::lerp(dx, co(0, 0, 0), co(1, 0, 0)),
                math::lerp(dx, co(0, 1, 0), co(1, 1, 0))),
            math::lerp(dy, math::lerp(dx, co(0, 0, 1), co(1, 0, 1)),
                math::lerp(dx, co(0, 1, 1), co(1, 1, 1))));
    }

    return RGBSigmoidPolynomial(c[0], c[1], c[2]);
}

extern const int sRGBToSpectrumTable_Res;
extern const float sRGBToSpectrumTable_Scale[64];
extern const RGBToSpectrumTable::CoefficientArray sRGBToSpectrumTable_Data;

RGBToSpectrumTable* RGBToSpectrumTable::sRGB = nullptr;

extern const int DCI_P3ToSpectrumTable_Res;
extern const float DCI_P3ToSpectrumTable_Scale[64];
extern const RGBToSpectrumTable::CoefficientArray DCI_P3ToSpectrumTable_Data;

RGBToSpectrumTable* RGBToSpectrumTable::DCI_P3 = nullptr;

extern const int REC2020ToSpectrumTable_Res;
extern const float REC2020ToSpectrumTable_Scale[64];
extern const RGBToSpectrumTable::CoefficientArray REC2020ToSpectrumTable_Data;

RGBToSpectrumTable* RGBToSpectrumTable::Rec2020 = nullptr;

extern const int ACES2065_1ToSpectrumTable_Res;
extern const float ACES2065_1ToSpectrumTable_Scale[64];
extern const RGBToSpectrumTable::CoefficientArray ACES2065_1ToSpectrumTable_Data;

RGBToSpectrumTable* RGBToSpectrumTable::ACES2065_1 = nullptr;

void RGBToSpectrumTable::init(Allocator alloc)
{
    float* sRGBToSpectrumTableScalePtr =
        (float*)alloc.allocate_bytes(sizeof(sRGBToSpectrumTable_Scale), alignof(float));
    memcpy(sRGBToSpectrumTableScalePtr, sRGBToSpectrumTable_Scale,
        sizeof(sRGBToSpectrumTable_Scale));
    RGBToSpectrumTable::CoefficientArray* sRGBToSpectrumTableDataPtr =
        (RGBToSpectrumTable::CoefficientArray*)alloc.allocate_bytes(
            sizeof(RGBToSpectrumTable::CoefficientArray), alignof(float));
    memcpy(sRGBToSpectrumTableDataPtr, sRGBToSpectrumTable_Data,
        sizeof(sRGBToSpectrumTable_Data));

    sRGB = alloc.new_object<RGBToSpectrumTable>(sRGBToSpectrumTableScalePtr,
        sRGBToSpectrumTableDataPtr);

    // DCI_P3
    float* DCI_P3ToSpectrumTableScalePtr =
        (float*)alloc.allocate_bytes(sizeof(DCI_P3ToSpectrumTable_Scale), alignof(float));
    memcpy(DCI_P3ToSpectrumTableScalePtr, DCI_P3ToSpectrumTable_Scale,
        sizeof(DCI_P3ToSpectrumTable_Scale));
    RGBToSpectrumTable::CoefficientArray* DCI_P3ToSpectrumTableDataPtr =
        (RGBToSpectrumTable::CoefficientArray*)alloc.allocate_bytes(
            sizeof(DCI_P3ToSpectrumTable_Data), alignof(float));
    memcpy(DCI_P3ToSpectrumTableDataPtr, DCI_P3ToSpectrumTable_Data,
        sizeof(DCI_P3ToSpectrumTable_Data));

    DCI_P3 = alloc.new_object<RGBToSpectrumTable>(DCI_P3ToSpectrumTableScalePtr,
        DCI_P3ToSpectrumTableDataPtr);

    // Rec2020
    float* REC2020ToSpectrumTableScalePtr =
        (float*)alloc.allocate_bytes(sizeof(REC2020ToSpectrumTable_Scale), alignof(float));
    memcpy(REC2020ToSpectrumTableScalePtr, REC2020ToSpectrumTable_Scale,
        sizeof(REC2020ToSpectrumTable_Scale));
    RGBToSpectrumTable::CoefficientArray* REC2020ToSpectrumTableDataPtr =
        (RGBToSpectrumTable::CoefficientArray*)alloc.allocate_bytes(
            sizeof(REC2020ToSpectrumTable_Data), alignof(float));
    memcpy(REC2020ToSpectrumTableDataPtr, REC2020ToSpectrumTable_Data,
        sizeof(REC2020ToSpectrumTable_Data));

    Rec2020 = alloc.new_object<RGBToSpectrumTable>(REC2020ToSpectrumTableScalePtr,
        REC2020ToSpectrumTableDataPtr);

    // ACES2065_1
    float* ACES2065_1ToSpectrumTableScalePtr =
        (float*)alloc.allocate_bytes(sizeof(ACES2065_1ToSpectrumTable_Scale), alignof(float));
    memcpy(ACES2065_1ToSpectrumTableScalePtr, ACES2065_1ToSpectrumTable_Scale,
        sizeof(ACES2065_1ToSpectrumTable_Scale));
    RGBToSpectrumTable::CoefficientArray* ACES2065_1ToSpectrumTableDataPtr =
        (RGBToSpectrumTable::CoefficientArray*)alloc.allocate_bytes(
            sizeof(ACES2065_1ToSpectrumTable_Data), alignof(float));
    memcpy(ACES2065_1ToSpectrumTableDataPtr, ACES2065_1ToSpectrumTable_Data,
        sizeof(ACES2065_1ToSpectrumTable_Data));

    ACES2065_1 = alloc.new_object<RGBToSpectrumTable>(
        ACES2065_1ToSpectrumTableScalePtr, ACES2065_1ToSpectrumTableDataPtr);

    checkCUDAError("RGBToSpectrumTable init");
}

RGBColorSpace::RGBColorSpace(const glm::vec2& r, const glm::vec2& g, const glm::vec2& b, SpectrumPtr illuminant, const RGBToSpectrumTable* rgbToSpectrumTable, Allocator alloc)
    :r(r), g(g), b(b), illuminant(illuminant, alloc), rgbToSpectrumTable(rgbToSpectrumTable)
{
    glm::vec3 W = spec::spectrum_to_xyz(illuminant);
    glm::vec3 R = xyY_to_XYZ(r), G = xyY_to_XYZ(g), B = xyY_to_XYZ(b);
    glm::vec2 w = glm::vec2(W.x,W.y);
    glm::mat3 rgb;
    rgb[0][0] = R.x;
    rgb[0][1] = R.y;
    rgb[0][2] = R.z;
    rgb[1][0] = G.x;
    rgb[1][1] = G.y;
    rgb[1][2] = G.z;
    rgb[2][0] = B.x;
    rgb[2][1] = B.y;
    rgb[2][2] = B.z;

    glm::vec3 C = glm::inverse(rgb) * W;
    XYZFromRGB = rgb * glm::diagonal3x3(C);
    RGBFromXYZ = glm::inverse(XYZFromRGB);
}

RGBSigmoidPolynomial RGBColorSpace::to_rgb_coeffs(const glm::vec3& rgb) const {
    return (*rgbToSpectrumTable)(glm::max(rgb, glm::vec3(0.0f)));
}

__device__ const RGBColorSpace* RGBColorSpace_sRGB;
__device__ const RGBColorSpace* RGBColorSpace_DCI_P3;
__device__ const RGBColorSpace* RGBColorSpace_Rec2020;
__device__ const RGBColorSpace* RGBColorSpace_ACES2065_1;

RGBColorSpace* RGBColorSpace::sRGB = nullptr;
RGBColorSpace* RGBColorSpace::DCI_P3 = nullptr;
RGBColorSpace* RGBColorSpace::Rec2020 = nullptr;
RGBColorSpace* RGBColorSpace::ACES2065_1 = nullptr;


void RGBColorSpace::init(Allocator alloc)
{
    // Rec. ITU-R BT.709.3
    sRGB = alloc.new_object<RGBColorSpace>(
        glm::vec2(.64, .33), glm::vec2(.3, .6), glm::vec2(.15, .06),
        spec::get_named_spectrum("stdillum-D65"), RGBToSpectrumTable::sRGB, alloc);
    // P3-D65 (display)
    DCI_P3 = alloc.new_object<RGBColorSpace>(
        glm::vec2(.68, .32), glm::vec2(.265, .690), glm::vec2(.15, .06),
        spec::get_named_spectrum("stdillum-D65"), RGBToSpectrumTable::DCI_P3, alloc);
    // ITU-R Rec BT.2020
    Rec2020 = alloc.new_object<RGBColorSpace>(
        glm::vec2(.708, .292), glm::vec2(.170, .797), glm::vec2(.131, .046),
        spec::get_named_spectrum("stdillum-D65"), RGBToSpectrumTable::Rec2020, alloc);
    ACES2065_1 = alloc.new_object<RGBColorSpace>(
        glm::vec2(.7347, .2653), glm::vec2(0., 1.), glm::vec2(.0001, -.077),
        spec::get_named_spectrum("illum-acesD60"), RGBToSpectrumTable::ACES2065_1, alloc);

    cudaMemcpyToSymbol(RGBColorSpace_sRGB, &RGBColorSpace::sRGB,
        sizeof(RGBColorSpace_sRGB));
    cudaMemcpyToSymbol(RGBColorSpace_DCI_P3, &RGBColorSpace::DCI_P3,
        sizeof(RGBColorSpace_DCI_P3));
    cudaMemcpyToSymbol(RGBColorSpace_Rec2020, &RGBColorSpace::Rec2020,
        sizeof(RGBColorSpace_Rec2020));
    cudaMemcpyToSymbol(RGBColorSpace_ACES2065_1,
        &RGBColorSpace::ACES2065_1,
        sizeof(RGBColorSpace_ACES2065_1));
    checkCUDAError("RGBColorSpace init");
}