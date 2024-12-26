#pragma once

#include "spectrum.h"
#include "glm/glm.hpp"
#include "memoryUtils.h"
#include "glm/gtx/matrix_operation.hpp"

__device__ __host__
inline glm::vec3 xyY_to_XYZ(const glm::vec2& xy, float Y = 1)
{
	if (xy.y == 0)
		return glm::vec3(0);
	return glm::vec3(xy.x * Y / xy.y, Y, (1 - xy.x - xy.y) * Y / xy.y);
}

__device__ __host__ 
inline glm::vec2 XYZ_to_xy(const glm::vec3& xyz)
{
	return { xyz.x / (xyz.x + xyz.y + xyz.z),xyz.y / (xyz.x + xyz.y + xyz.z) };
}

class RGBSigmoidPolynomial
{
public:
	RGBSigmoidPolynomial() = default;
	__device__ __host__
		RGBSigmoidPolynomial(float c0, float c1, float c2) : c0(c0), c1(c1), c2(c2) {}

	__device__ __host__
		float operator()(float lambda) const {
		return s(math::evaluate_polynomial(lambda, c2, c1, c0));
	}

	__device__ __host__
		float max_value() const {
		float result = std::max((*this)(360), (*this)(830));
		float lambda = -c1 / (2 * c0);
		if (lambda >= 360 && lambda <= 830)
			result = std::max(result, (*this)(lambda));
		return result;
	}

private:
	__device__ __host__
		static float s(float x) {
		if (isinf(x))
			return x > 0 ? 1 : 0;
		return .5f + x / (2 * std::sqrt(1 + math::sqr(x)));
	};

	float c0, c1, c2;
};

class RGBToSpectrumTable
{
public:
	static constexpr int res = 64;

	using CoefficientArray = float[3][res][res][res][3];

	RGBToSpectrumTable(const float* zNodes, const CoefficientArray* coeffs)
		: zNodes(zNodes), coeffs(coeffs) {}


	__device__ __host__ RGBSigmoidPolynomial operator()(const glm::vec3& rgb) const;

	static void init(Allocator alloc);

	static RGBToSpectrumTable* sRGB;
	static RGBToSpectrumTable* DCI_P3;
	static RGBToSpectrumTable* Rec2020;
	static RGBToSpectrumTable* ACES2065_1;

private:
	const float* zNodes;
	const CoefficientArray* coeffs;
};

class RGBColorSpace
{
public:
	RGBColorSpace(const glm::vec2& r, const glm::vec2& g, const glm::vec2& b, SpectrumPtr illuminant, const RGBToSpectrumTable* rgbToSpectrumTable, Allocator alloc)
		;

	__device__ __host__
		RGBSigmoidPolynomial to_rgb_coeffs(const glm::vec3& rgb) const;
	static void init(Allocator alloc);
	glm::vec2 r, g, b, w;
	DenselySampledSpectrum illuminant;
	glm::mat3 XYZFromRGB, RGBFromXYZ;
	__device__ __host__ glm::vec3 to_rgb(const glm::vec3& xyz) const { return RGBFromXYZ * xyz; }
	__device__ __host__ glm::vec3 to_xyz(const glm::vec3& rgb) const { return XYZFromRGB * rgb; }
	static RGBColorSpace* sRGB, * DCI_P3, * Rec2020, * ACES2065_1;

	__device__ __host__
		glm::vec3 LuminanceVector() const {
		return glm::vec3(XYZFromRGB[0][1], XYZFromRGB[1][1], XYZFromRGB[2][1]);
	}
private:
	const RGBToSpectrumTable* rgbToSpectrumTable;
};




class RGBAlbedoSpectrum {
public:
	__device__ __host__
		float operator()(float lambda) const { return rsp(lambda); }
	__device__ __host__
		float max_value() const { return rsp.max_value(); }

	__device__ __host__
		SampledSpectrum	sample(const SampledWavelengths& swl) const
	{
		SampledSpectrum s;
		for (int i = 0; i < spec::NSpectrumSamples; ++i)
			s[i] = rsp(swl[i]);
		return s;
	}

	__device__ __host__
		RGBAlbedoSpectrum(const RGBColorSpace& cs, const glm::vec3& rgb)
	{
		rsp = cs.to_rgb_coeffs(rgb);
	}
private:
	RGBSigmoidPolynomial rsp;
};

class RGBUnboundedSpectrum {
public:
	__device__ __host__
		float operator()(float lambda) const { return scale * rsp(lambda); }
	__device__ __host__
		float max_value() const { return scale * rsp.max_value(); }

	__device__ __host__
		SampledSpectrum	sample(const SampledWavelengths& swl) const
	{
		SampledSpectrum s;
		for (int i = 0; i < spec::NSpectrumSamples; ++i)
			s[i] = scale * rsp(swl[i]);
		return s;
	}
	__device__ __host__
		RGBUnboundedSpectrum(const RGBColorSpace& cs, const glm::vec3& rgb)
	{
		float m = std::max({ rgb.r, rgb.g, rgb.b });
		scale = 2 * m;
		rsp = cs.to_rgb_coeffs(scale ? rgb / scale : glm::vec3(0, 0, 0));
	}
	__device__ __host__
		RGBUnboundedSpectrum() : rsp(0, 0, 0), scale(0) {}
private:
	float scale = 1.0f;
	RGBSigmoidPolynomial rsp;
};

class RGBIlluminantSpectrum {
public:
	RGBIlluminantSpectrum() = default;
	__device__ __host__
		RGBIlluminantSpectrum(const RGBColorSpace& cs, const glm::vec3& rgb) : illuminant(&cs.illuminant)
	{
		float m = std::max({ rgb.r, rgb.g, rgb.b });
		scale = 2 * m;
		rsp = cs.to_rgb_coeffs(scale ? rgb / scale : glm::vec3(0, 0, 0));
	}
	__device__ __host__
		float operator()(float lambda) const {
		if (!illuminant)
			return 0;
		return scale * rsp(lambda) * (*illuminant)(lambda);
	}

	__device__ __host__
		float max_value() const {
		if (!illuminant)
			return 0;
		return scale * rsp.max_value() * illuminant->max_value();
	}
	__device__ __host__
		SampledSpectrum sample(const SampledWavelengths& lambda) const {
		if (!illuminant)
			return SampledSpectrum(0.0f);
		SampledSpectrum s;
		for (int i = 0; i < spec::NSpectrumSamples; ++i)
			s[i] = scale * rsp(lambda[i]);
		return s * illuminant->sample(lambda);
	}

private:
	float scale;
	RGBSigmoidPolynomial rsp;
	const DenselySampledSpectrum* illuminant = nullptr;
};

__device__ extern const RGBColorSpace* RGBColorSpace_sRGB;
__device__ extern const RGBColorSpace* RGBColorSpace_DCI_P3;
__device__ extern const RGBColorSpace* RGBColorSpace_Rec2020;
__device__ extern const RGBColorSpace* RGBColorSpace_ACES2065_1;


// White Balance Definitions
// clang-format off
// These are the Bradford transformation matrices.
const glm::mat3 LMSFromXYZ(0.8951, 0.2664, -0.1614,
	-0.7502, 1.7135, 0.0367,
	0.0389, -0.0685, 1.0296);
const glm::mat3 XYZFromLMS(0.986993, -0.147054, 0.159963,
	0.432305, 0.51836, 0.0492912,
	-0.00852866, 0.0400428, 0.968487);
// clang-format on

inline glm::mat3 white_balance(glm::vec2 srcWhite, glm::vec2 targetWhite) {
	// Find LMS coefficients for source and target white
	glm::vec3 srcXYZ = xyY_to_XYZ(srcWhite), dstXYZ = xyY_to_XYZ(targetWhite);
	auto srcLMS = LMSFromXYZ * srcXYZ, dstLMS = LMSFromXYZ * dstXYZ;

	// Return white balancing matrix for source and target white
	glm::mat3 LMScorrect;
	LMScorrect[0][0] = dstLMS[0] / srcLMS[0];
	LMScorrect[1][1] = dstLMS[1] / srcLMS[1];
	LMScorrect[2][2] = dstLMS[2] / srcLMS[2];
	return XYZFromLMS * LMScorrect * LMSFromXYZ;
}