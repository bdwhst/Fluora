#include "spectrum.h"
#include "color.h"
#include <map>
#include <string>
#include "utilities.h"
#include "SpectrumConsts/spectrum_data.h"

__device__ __host__  float SpectrumPtr::operator()(float lambda) const
{
    auto op = [&](auto ptr) { return (*ptr)(lambda); };
    return Dispatch(op);
}

__device__ __host__  float SpectrumPtr::max_value() const
{
    auto op = [&](auto ptr) {return ptr->max_value(); };
    return Dispatch(op);
}

__device__ __host__  SampledSpectrum SpectrumPtr::sample(const SampledWavelengths& lambda) const
{
    auto op = [&](auto ptr) {return ptr->sample(lambda); };
    return Dispatch(op);
}

PiecewiseLinearSpectrum* PiecewiseLinearSpectrum::from_interleaved(const float* samples, size_t length, bool normalize, Allocator alloc)
{
	std::vector<float> lambdas, values;
	//cheap but useful hack
	if (samples[0] > spec::gLambdaMin)
	{
		lambdas.push_back(spec::gLambdaMin - 1);
		values.push_back(samples[1]);
	}
	for (size_t i = 0; i < length; i++)
	{
		lambdas.push_back(samples[i << 1]);
		values.push_back(samples[(i << 1) + 1]);
	}
	if (lambdas.back() < spec::gLambdaMax)
	{
		lambdas.push_back(spec::gLambdaMax + 1);
		values.push_back(values.back());
	}
	PiecewiseLinearSpectrum* s = alloc.new_object<PiecewiseLinearSpectrum>(lambdas.size(), lambdas.data(), values.data(), alloc);

    if (normalize)
    {
        s->scale(spec::CIE_Y_integral / inner_product(s, &spec::Y()));
    }
    return s;
}




namespace spec
{
    glm::vec3 spectrum_to_xyz(SpectrumPtr s) {
        return glm::vec3(inner_product(&spec::X(), s), inner_product(&spec::Y(), s),
            inner_product(&spec::Z(), s)) /
            spec::CIE_Y_integral;
    }

    DenselySampledSpectrum D(float temperature, Allocator alloc) {
        // Convert temperature to CCT
        float cct = temperature * 1.4388f / 1.4380f;
        if (cct < 4000) {
            // CIE D ill-defined, use blackbody
            BlackbodySpectrum bb = BlackbodySpectrum(cct);
            DenselySampledSpectrum blackbody = DenselySampledSpectrum::sample_function(
                [=](float lambda) { return bb(lambda); });

            return blackbody;
        }

        // Convert CCT to xy
        float x;
        if (cct <= 7000)
            x = -4.607f * 1e9f / math::pow<3>(cct) + 2.9678f * 1e6f / math::sqr(cct) +
            0.09911f * 1e3f / cct + 0.244063f;
        else
            x = -2.0064f * 1e9f / math::pow<3>(cct) + 1.9018f * 1e6f / math::sqr(cct) +
            0.24748f * 1e3f / cct + 0.23704f;
        float y = -3 * x * x + 2.870f * x - 0.275f;

        // Interpolate D spectrum
        float M = 0.0241f + 0.2562f * x - 0.7341f * y;
        float M1 = (-1.3515f - 1.7703f * x + 5.9114f * y) / M;
        float M2 = (0.0300f - 31.4424f * x + 30.0717f * y) / M;

        std::vector<float> values(nCIES);
        for (int i = 0; i < nCIES; ++i)
            values[i] = (CIE_S0[i] + CIE_S1[i] * M1 + CIE_S2[i] * M2) * 0.01;

        PiecewiseLinearSpectrum dpls(nCIES, (float*)CIE_S_lambda, values.data());
        return DenselySampledSpectrum(&dpls, alloc);
    }
}

