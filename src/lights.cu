#include "lights.h"

__device__ SampledSpectrum LightPtr::phi(SampledWavelengths lambda) const
{
	auto f = [&](auto ptr)
		{
			return ptr->phi(lambda);
		};
	return Dispatch(f);
}

__device__ LightType LightPtr::type() const
{
	auto f = [&](auto ptr)
		{
			return ptr->type();
		};
	return Dispatch(f);
}

__device__ LightLiSample LightPtr::sample_Li(const LightSampleContext& ctx, glm::vec3 rand, SampledWavelengths lambda, bool allowIncompletePDF) const
{
	auto f = [&](auto ptr)
		{
			return ptr->sample_Li(ctx,rand,lambda,allowIncompletePDF);
		};
	return Dispatch(f);
}

__device__ float LightPtr::pdf_Li(const LightSampleContext& ctx, const glm::vec3& pLight, const glm::vec3& nLight, bool allowIncompletePDF)
{
	auto f = [&](auto ptr)
		{
			return ptr->pdf_Li(ctx, pLight, nLight, allowIncompletePDF);
		};
	return Dispatch(f);
}

__device__ SampledSpectrum LightPtr::L(const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv, const glm::vec3& w, const SampledWavelengths& lambda) const
{
	auto f = [&](auto ptr)
		{
			return ptr->L(p,n,uv,w,lambda);
		};
	return Dispatch(f);
}

__device__ SampledSpectrum LightPtr::Le(const Ray& ray, const SampledWavelengths& lambda) const
{
	auto f = [&](auto ptr)
		{
			return ptr->Le(ray, lambda);
		};
	return Dispatch(f);
}

LightBase::SpectrumInstanceSet* LightBase::m_spectrumInstanceSet = nullptr;

