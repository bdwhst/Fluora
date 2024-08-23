#pragma once
#include "taggedptr.h"
#include "mathUtils.h"
#include "memoryUtils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <span>
namespace spec
{
	constexpr float gLambdaMin = 360, gLambdaMax = 830;

	static constexpr int NSpectrumSamples = 4;

	static constexpr float CIE_Y_integral = 106.856895;

	void init(Allocator alloc);
}

class SampledSpectrum
{
public:
	__device__ __host__ SampledSpectrum() = default;
	__device__ __host__ explicit SampledSpectrum(float* v)
	{
		for (int i = 0; i < spec::NSpectrumSamples; ++i)
			values[i] = v[i];
	}
	__device__ __host__ explicit SampledSpectrum(float v)
	{
		for (int i = 0; i < spec::NSpectrumSamples; ++i)
			values[i] = v;
	}
	__device__ __host__ SampledSpectrum& operator+=(const SampledSpectrum& s)
	{
		for (int i = 0; i < spec::NSpectrumSamples; i++)
		{
			values[i] += s.values[i];
		}
		return *this;
	}
	__device__ __host__ SampledSpectrum& operator-=(const SampledSpectrum& s)
	{
		for (int i = 0; i < spec::NSpectrumSamples; i++)
		{
			values[i] -= s.values[i];
		}
		return *this;
	}
	__device__ __host__ SampledSpectrum& operator*=(const SampledSpectrum& s)
	{
		for (int i = 0; i < spec::NSpectrumSamples; i++)
		{
			values[i] *= s.values[i];
		}
		return *this;
	}
	__device__ __host__ SampledSpectrum& operator/=(const SampledSpectrum& s)
	{
		for (int i = 0; i < spec::NSpectrumSamples; i++)
		{
			values[i] /= s.values[i];
		}
		return *this;
	}

	__device__ __host__ SampledSpectrum operator+(const SampledSpectrum& s) const
	{
		SampledSpectrum ans = *this;
		return ans += s;
	}
	__device__ __host__ SampledSpectrum operator-(const SampledSpectrum& s) const
	{
		SampledSpectrum ans = *this;
		return ans -= s;
	}
	__device__ __host__ SampledSpectrum operator*(const SampledSpectrum& s) const
	{
		SampledSpectrum ans = *this;
		return ans *= s;
	}
	__device__ __host__ SampledSpectrum operator/(const SampledSpectrum& s) const
	{
		SampledSpectrum ans = *this;
		return ans /= s;
	}

	__device__ __host__ SampledSpectrum& operator*=(float a)
	{
		for (int i = 0; i < spec::NSpectrumSamples; ++i)
			values[i] *= a;
		return *this;
	}

	__device__ __host__ SampledSpectrum operator*(float a) const
	{
		SampledSpectrum ans = *this;
		ans *= a;
		return ans;
	}

	__device__ __host__ SampledSpectrum operator/(float a) const
	{
		SampledSpectrum ans = *this;
		for (int i = 0; i < spec::NSpectrumSamples; ++i)
			ans.values[i] /= a;
		return ans;
	}

	__device__ __host__ float operator==(const SampledSpectrum& s)
	{
		bool eq = values[0] == s.values[0];
		for (int i = 1; i < spec::NSpectrumSamples; i++)
		{
			eq &= values[i] == s.values[i];
		}
		return eq;
	}

	__device__ __host__ float operator!=(const SampledSpectrum& s)
	{
		bool ieq = values[0] != s.values[0];
		for (int i = 1; i < spec::NSpectrumSamples; i++)
		{
			ieq |= values[i] != s.values[i];
		}
		return ieq;
	}

	__device__ __host__ float operator[](int i) const
	{
		return values[i];
	}

	__device__ __host__ float& operator[](int i)
	{
		return values[i];
	}

	__device__ __host__ float average() const {
		float ans = values[0];
		for (int i = 1; i < spec::NSpectrumSamples; i++)
		{
			ans += values[i];
		}
		return ans / spec::NSpectrumSamples;
	}

	__device__ __host__ bool is_nan() const
	{
		bool ans = false;
		for (int i = 0; i < spec::NSpectrumSamples; i++)
		{
			ans = ans || math::is_nan(values[i]);
		}
		return ans;
	}

	float values[spec::NSpectrumSamples];
};

__device__ __host__ inline SampledSpectrum operator*(float a, const SampledSpectrum& s) { return s * a; }

__device__ __host__ inline SampledSpectrum safe_div(const SampledSpectrum& s0, const SampledSpectrum& s1)
{
	SampledSpectrum s;
	for (int i = 0; i < spec::NSpectrumSamples; i++)
	{
		s[i] = s1[i] == 0 ? s0[i] : s0[i] / s1[i];
	}
	return s;
}


class SampledWavelengths
{
public:
	__device__ __host__ bool operator==(const SampledWavelengths& swl)
	{
		bool eq0 = m_lambda[0] == swl.m_lambda[0];
		bool eq1 = m_pdf[1] == swl.m_pdf[1];
		for (int i = 1; i < spec::NSpectrumSamples; i++)
		{
			eq0 &= m_lambda[i] == swl.m_lambda[i];
			eq1 &= m_pdf[i] == swl.m_pdf[i];
		}
		return eq0 && eq1;
	}

	__device__ __host__ bool operator!=(const SampledWavelengths& swl)
	{
		bool ieq0 = m_lambda[0] != swl.m_lambda[0];
		bool ieq1 = m_pdf[1] != swl.m_pdf[1];
		for (int i = 1; i < spec::NSpectrumSamples; i++)
		{
			ieq0 |= m_lambda[i] != swl.m_lambda[i];
			ieq1 |= m_pdf[i] != swl.m_pdf[i];
		}
		return ieq0 || ieq1;
	}
	__device__ __host__ static SampledWavelengths sample_uniform(float u, float lmin = spec::gLambdaMin, float lmax = spec::gLambdaMax)
	{
		SampledWavelengths swl;
		swl.m_lambda[0] = math::lerp(u, lmin, lmax);
		float step = (lmax - lmin) / spec::NSpectrumSamples;
		for (int i = 1; i < spec::NSpectrumSamples; i++)
		{
			swl.m_lambda[i] = swl.m_lambda[i - 1] + step;
			if (swl.m_lambda[i] > lmax)
			{
				swl.m_lambda[i] += (lmin - lmax);
			}
		}
		float pdf_uniform = 1 / (lmax - lmin);
		for (int i = 0; i < spec::NSpectrumSamples; i++)
		{
			swl.m_pdf[i] = pdf_uniform;
		}
		return swl;

	}

	__device__ __host__ static SampledWavelengths sample_visible(float u)
	{
		SampledWavelengths swl;
		for (int i = 0; i < spec::NSpectrumSamples; i++)
		{
			float up = u+(float)i / spec::NSpectrumSamples;
			if (up > 1) up -= 1;
			swl.m_lambda[i] = SampleVisibleWavelengths(up);
			swl.m_pdf[i] = VisibleWavelengthsPDF(swl.m_lambda[i]);
		}
		return swl;
	}

	__device__ __host__ bool secondary_terminated() const
	{
		for (int i = 1; i < spec::NSpectrumSamples; i++)
			if (m_pdf[i] != 0) return false;
		return true;
	}
	__device__ __host__ void terminate_secondary()
	{
		if (secondary_terminated()) return;
		for (int i = 1; i < spec::NSpectrumSamples; i++)
			m_pdf[i] = 0;
		m_pdf[0] /= spec::NSpectrumSamples;
	}
	__device__ __host__ SampledSpectrum pdf() const
	{
		return SampledSpectrum((float*)m_pdf);
	}
	__device__ __host__ float operator[](int i) const { return m_lambda[i]; }
	__device__ __host__ float& operator[](int i) { return m_lambda[i]; }
	float m_lambda[spec::NSpectrumSamples];
	float m_pdf[spec::NSpectrumSamples];
private:
	__device__ __host__ static float SampleVisibleWavelengths(float u) {
		return 538 - 138.888889f * atanhf(0.85691062f - 1.82750197f * u);
	}

	__device__ __host__ static float VisibleWavelengthsPDF(float lambda) {
		if (lambda < 360 || lambda > 830)
			return 0;
		return 0.0039398042f / math::sqr(coshf(0.0072f * (lambda - 538)));
	}
};


class ConstantSpectrum;
class PiecewiseLinearSpectrum;
class DenselySampledSpectrum;
class RGBAlbedoSpectrum;
class RGBUnboundedSpectrum;
class RGBIlluminantSpectrum;

class Spectrum :public TaggedPointer< ConstantSpectrum, PiecewiseLinearSpectrum, DenselySampledSpectrum, RGBAlbedoSpectrum, RGBUnboundedSpectrum, RGBIlluminantSpectrum>
{
public:
	using TaggedPointer::TaggedPointer;
	__device__ __host__ float operator()(float lambda) const;
	__device__ __host__ float max_value() const;
	__device__ __host__ SampledSpectrum sample(const SampledWavelengths& lambda) const;
};

inline float inner_product(Spectrum f, Spectrum g) {
	float integral = 0;
	for (float lambda = spec::gLambdaMin; lambda <= spec::gLambdaMax; ++lambda)
		integral += f(lambda) * g(lambda);
	return integral;
}

namespace spec
{
	glm::vec3 spectrum_to_xyz(Spectrum s);
	Spectrum get_named_spectrum(const std::string& name);
}



class ConstantSpectrum
{
	float c;
public:
	__device__ __host__ ConstantSpectrum(float c):c(c){}
	__device__ __host__ float operator()(float lamdba) const { return c; }
	__device__ __host__ SampledSpectrum sample(const SampledWavelengths& swl) const {
		return SampledSpectrum(c);
	}
	__device__ __host__ float max_value() const { return c; }
};

class DenselySampledSpectrum
{
	int32_t lambda_min = spec::gLambdaMin, lambda_max = spec::gLambdaMax;
	float* values = nullptr;
	Allocator alloc;
	friend struct std::hash<DenselySampledSpectrum>;
public:
	DenselySampledSpectrum(int lmin, int lmax, Allocator alloc={}):lambda_min(lmin),lambda_max(lmax),alloc(alloc)
	{
		values = alloc.allocate<float>(get_size());
	}
	DenselySampledSpectrum(Spectrum spec, int lmin, int lmax, Allocator  alloc = {}) :lambda_min(lmin), lambda_max(lmax), alloc(alloc)
	{
		values = alloc.allocate<float>(get_size());
		if (spec)
		{
			for (int l = lmin; l <= lmax; l++)
			{
				values[l-lmin] = spec(l);
			}
		}
	}
	DenselySampledSpectrum(Spectrum s, Allocator alloc)
		: DenselySampledSpectrum(s, spec::gLambdaMin, spec::gLambdaMax, alloc) {}
	~DenselySampledSpectrum()
	{
		alloc.deallocate(values, get_size());
	}

	template <typename F>
	static DenselySampledSpectrum sample_function(F func, int lmin = spec::gLambdaMin, int lmax = spec::gLambdaMax, Allocator alloc = {})
	{
		DenselySampledSpectrum s(lmin, lmax, alloc);
		for (int l = lmin; l <= lmax; l++)
		{
			s.values[l - lmin] = func(l);
		}
		return s;
	}

	__device__ __host__ SampledSpectrum sample(const SampledWavelengths& swl) const
	{
		SampledSpectrum s;
		for (int i = 0; i < spec::NSpectrumSamples; i++)
		{
			int offset = std::lround(swl[i]) - lambda_min;
			if (offset < 0 || offset >= get_size())
			{
				s[i] = 0;
			}
			else
			{
				s[i] = values[offset];
			}
		}
		return s;
	}

	__device__ __host__ float operator()(float lambda) const 
	{
		int offset = std::lround(lambda) - lambda_min;
		if (offset < 0 || offset >= get_size())
			return 0;
		return values[offset];
	}

	__device__ __host__ bool operator==(const DenselySampledSpectrum& anotherS)
	{
		if (lambda_min != anotherS.lambda_min || lambda_max != anotherS.lambda_max) return false;
		for (int32_t i = lambda_min; i <= lambda_max; i++)
		{
			if (values[i] != anotherS.values[i]) return false;
		}
		return true;
	}

	__device__ __host__ uint32_t get_size() const 
	{
		return  lambda_max - lambda_min + 1;
	}

	__device__ __host__ float max_value() const { return *std::max_element(values, values + get_size()); }
};

template <>
struct std::hash<DenselySampledSpectrum> {
	__device__ __host__
		size_t operator()(const DenselySampledSpectrum& s) const {
		return math::HashBuffer(s.values, sizeof(float) * s.get_size());
	}
};

namespace spec
{
	extern __device__ DenselySampledSpectrum* dev_x, * dev_y, * dev_z;
	extern DenselySampledSpectrum* x, * y, * z;
	__host__ __device__ inline const DenselySampledSpectrum& X() {
#if defined(__CUDA_ARCH__)
		return *dev_x;
#else
		return *x;
#endif
	}

	__host__ __device__ inline const DenselySampledSpectrum& Y() {
#if defined(__CUDA_ARCH__)
		return *dev_y;
#else
		return *y;
#endif
	}

	__host__ __device__ inline const DenselySampledSpectrum& Z() {
#if defined(__CUDA_ARCH__)
		return *dev_z;
#else
		return *z;
#endif
	}

}

class PiecewiseLinearSpectrum {
	float* lambdas = nullptr, * values = nullptr;
	int length = 0;
	Allocator alloc;
public:
	PiecewiseLinearSpectrum() = default;
	PiecewiseLinearSpectrum(const PiecewiseLinearSpectrum& s) = delete;
	PiecewiseLinearSpectrum(int length, float* lambdas, float* values, Allocator alloc = {}):length(length),alloc(alloc)
	{
		size_t alloc_size = sizeof(float) * length;
		this->lambdas = alloc.allocate<float>(length);
		this->values = alloc.allocate<float>(length);
		cudaMemcpy(this->lambdas, lambdas, alloc_size, cudaMemcpyDefault);
		cudaMemcpy(this->values, values, alloc_size, cudaMemcpyDefault);
	}
	~PiecewiseLinearSpectrum()
	{
		alloc.deallocate(lambdas, length);
		alloc.deallocate(values, length);
	}
	__device__ __host__ float max_value() const
	{
		if (length == 0) return 0;
		return *std::max_element(values, values + length);
	}
	__device__ __host__ float operator()(float lambda) const
	{
		if (!lambdas || lambda<lambdas[0] || lambda>lambdas[length - 1]) return 0;
		auto f = [&](size_t i) { return lambdas[i] <= lambda; };
		int i = math::find_interval(length, f);
		float a = (lambda - lambdas[i]) / (lambdas[i + 1] - lambdas[i]);
		return math::lerp(a, values[i], values[i + 1]);
	}
	__device__ __host__ SampledSpectrum sample(const SampledWavelengths& lambda) const
	{
		SampledSpectrum s;
		for (int i = 0; i < spec::NSpectrumSamples; i++)
		{
			s[i] = (*this)(lambda[i]);
		}
		return s;
	}

	__device__ __host__ void scale(float s)
	{
		for (int i = 0; i < length; i++)
		{
			values[i] *= s;
		}
	}

	static PiecewiseLinearSpectrum* from_interleaved(const float* samples, size_t length, bool normalize, Allocator alloc);
	
};

#define FLT_ARRAY_HALFSIZE(arr) (sizeof(arr)/sizeof(float)/2)
#define FROM_INTERLEAVED_FLT_ARRAY(arr, normalized, alloc) (PiecewiseLinearSpectrum::from_interleaved((float*)(arr), FLT_ARRAY_HALFSIZE(arr), normalized, alloc))

