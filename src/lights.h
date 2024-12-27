#pragma once

#include "light.h"
#include "intersections.h"
#include "sampling.h"
class DiffuseAreaLight : public LightBase
{
public:
	DiffuseAreaLight(int primID, bool twoSided, SpectrumPtr s, float scale=1.0f) :LightBase(LightType::Area), m_primitiveID(primID), m_twoSided(twoSided), m_Lemit(nullptr), m_scale(scale)
	{
		m_Lemit = lookup_spectrum(s);
	}
	static DiffuseAreaLight* create(const BundledParams& params, Allocator alloc)
	{
		int primitiveID = params.get_int("primitiveID");
		if (primitiveID == -1)
		{
			throw std::runtime_error("Failed to get primitiveID for DiffuseAreaLight");
		}
		SpectrumPtr spec = params.get_spec("Le_spec");
		if (!spec)
		{
			throw std::runtime_error("Failed to get Le_spec for DiffuseAreaLight");
		}
		bool twoSided = params.get_bool("twoSided");
		float scale = params.get_float("scale", 1.0f);
		return alloc.new_object<DiffuseAreaLight>(primitiveID, twoSided, spec, scale);
	}
	GPU_FUNC SampledSpectrum phi(SampledWavelengths lambda) const
	{
		//TODO
		assert(0);
		return {};
	}
	GPU_FUNC LightLiSample sample_Li(const LightSampleContext& ctx, glm::vec3 rand, SampledWavelengths lambda, bool allowIncompletePDF = false) const;
	GPU_FUNC float pdf_Li(const LightSampleContext& ctx, const glm::vec3& pLight, const glm::vec3& nLight, bool allowIncompletePDF = false) const;
	
	GPU_FUNC SampledSpectrum L(const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv, const glm::vec3& w, const SampledWavelengths& lambda) const
	{
		if (!m_twoSided && glm::dot(n, w) < 0)
			return SampledSpectrum(0.f);
		return m_scale * (*m_Lemit)(lambda);
	}
	GPU_FUNC SampledSpectrum Le(const Ray& ray, const SampledWavelengths& lambda) const
	{
		return {};
	}
private:
	int m_primitiveID;
	bool m_twoSided;
	const DenselySampledSpectrum* m_Lemit;
	float m_scale;
};

class ImageInfiniteLight :public LightBase
{
public:
	ImageInfiniteLight(cudaTextureObject_t texture, float scale, float* illumFunc, int width, int height, Allocator alloc):LightBase(LightType::Infinite), m_dist(nullptr), m_texture(texture), m_scale(scale)
	{
		m_dist = alloc.new_object<Distribution2D>(illumFunc, width, height, alloc);
	}
	static ImageInfiniteLight* create(const BundledParams& params, Allocator alloc)
	{
		cudaTextureObject_t textureHandle = params.get_texture("textureObject");
		float scale = params.get_float("scale", 1.0f);
		float* illumFunc = (float*)params.get_ptr("illumFunc", nullptr);
		if (illumFunc == nullptr)
			throw std::runtime_error("illumFunc is null");

		int width = params.get_int("width", 0);
		int height = params.get_int("height", 0);
		if (width == 0 || height == 0)
			throw std::runtime_error("width or height is 0");

		return alloc.new_object<ImageInfiniteLight>(textureHandle, scale, illumFunc, width, height, alloc);
	}
	GPU_FUNC SampledSpectrum phi(SampledWavelengths lambda) const
	{
		//TODO
		assert(0);
		return {};
	}
	GPU_FUNC LightLiSample sample_Li(const LightSampleContext& ctx, glm::vec3 rand, SampledWavelengths lambda, bool allowIncompletePDF = false) const;
	GPU_FUNC float pdf_Li(const LightSampleContext& ctx, const glm::vec3& pLight, const glm::vec3& nLight, bool allowIncompletePDF = false) const;

	GPU_FUNC SampledSpectrum L(const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv, const glm::vec3& w, const SampledWavelengths& lambda) const;
	GPU_FUNC SampledSpectrum Le(const Ray& ray, const SampledWavelengths& lambda) const
	{
		return {};
	}
private:
	Distribution2D* m_dist;
	cudaTextureObject_t m_texture;
	float m_scale;
};