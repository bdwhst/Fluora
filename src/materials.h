#pragma once
#include "taggedptr.h"
#include "bsdf.h"
#include "spectrum.h"
#include "color.h"
#include <unordered_map>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include "memoryUtils.h"
#include "containers.h"
class DiffuseMaterial;
class DielectricMaterial;
class ConductorMaterial;
class EmissiveMaterial;



struct MaterialEvalInfo
{
	glm::vec3 wo;
	glm::vec2 uv;
	SampledWavelengths& swl;
	__device__ MaterialEvalInfo(const glm::vec3& wo, const glm::vec2& uv, SampledWavelengths& swl) :wo(wo), uv(uv), swl(swl){}
};

class MaterialPtr : public TaggedPointer<DiffuseMaterial, DielectricMaterial, ConductorMaterial, EmissiveMaterial>
{
public:
	using TaggedPointer::TaggedPointer;
	static MaterialPtr create(const std::string& name, const BundledParams& params, Allocator alloc);
	
	// Assume local allocated size is greater than any of the bxdf class
	__device__ BxDFPtr get_bxdf(MaterialEvalInfo& info, void* localMem);
};

class DiffuseMaterial
{
public:
	static DiffuseMaterial* create(const BundledParams& params, Allocator alloc)
	{
		glm::vec3  albedo = params.get_vec3("albedo");
		cudaTextureObject_t albedoMap = params.get_texture("albedoMap");
		RGBColorSpace* colorSpace = (RGBColorSpace*)params.get_ptr("colorSpace");
		if (!colorSpace)
			throw std::runtime_error("No color space specified for DiffuseMaterial");
		return alloc.new_object<DiffuseMaterial>(albedo, albedoMap, colorSpace);
	}
	DiffuseMaterial(const glm::vec3& albedo, cudaTextureObject_t albedoMap, RGBColorSpace* colorSpace):albedo(albedo), albedoMap(albedoMap), colorSpace(colorSpace){}
	__device__ BxDFPtr get_bxdf(MaterialEvalInfo& info, void* localMem);
private:
	glm::vec3   albedo = glm::vec3(0.5f);
	cudaTextureObject_t albedoMap = 0;
	RGBColorSpace* colorSpace;
};

class DielectricMaterial
{
public:
	static DielectricMaterial* create(const BundledParams& params, Allocator alloc)
	{
		SpectrumPtr eta = params.get_spec("eta");
		return alloc.new_object<DielectricMaterial>(eta);
	}
	DielectricMaterial(SpectrumPtr eta):eta(eta){}
	__device__ BxDFPtr get_bxdf(MaterialEvalInfo& info, void* localMem)
	{
		DielectricBxDF* bxdfPtr = (DielectricBxDF*)localMem;
		if (!eta) return nullptr;
		float sampledEta = eta(info.swl[0]);
		sampledEta = info.wo.z < 0 ? 1.0 / sampledEta : sampledEta;
		if (!eta.template Is<ConstantSpectrum>())
		{
			info.swl.terminate_secondary();
		}
		bxdfPtr->eta = sampledEta;
		return bxdfPtr;
	}
private:
	SpectrumPtr eta;
};

class ConductorMaterial
{
public:
	static ConductorMaterial* create(const BundledParams& params, Allocator alloc)
	{
		SpectrumPtr eta = params.get_spec("eta");
		SpectrumPtr k = params.get_spec("k");
		float roughness = params.get_float("roughness");
		return alloc.new_object<ConductorMaterial>(eta, k, roughness);
	}
	ConductorMaterial(SpectrumPtr eta, SpectrumPtr k, float roughness):eta(eta),k(k), roughness(roughness){}
	__device__ BxDFPtr get_bxdf(MaterialEvalInfo& info, void* localMem)
	{
		ConductorBxDF* bxdfPtr = (ConductorBxDF*)localMem;
		if (!eta || !k) return nullptr;
		SampledSpectrum sampledEta = eta.sample(info.swl);
		SampledSpectrum sampledK = k.sample(info.swl);
		bxdfPtr->eta = sampledEta;
		bxdfPtr->k = sampledK;
		bxdfPtr->dist.alpha_x = roughness;
		bxdfPtr->dist.alpha_y = roughness;
		return bxdfPtr;
	}
private:
	SpectrumPtr eta, k;
	float roughness;
};

// Diffuse emissive material
class EmissiveMaterial
{
public:
	static EmissiveMaterial* create(const BundledParams& params, Allocator alloc)
	{
		glm::vec3 albedo = params.get_vec3("albedo");
		float emittance = params.get_float("emittance");
		RGBColorSpace* colorSpace = (RGBColorSpace*)params.get_ptr("colorSpace");
		if (!colorSpace)
			throw std::runtime_error("No color space specified for EmissiveMaterial");
		return alloc.new_object<EmissiveMaterial>(albedo * emittance, colorSpace);
	}
	EmissiveMaterial(const glm::vec3& rgb, RGBColorSpace* colorSpace):rgb(rgb), colorSpace(colorSpace){}
	//should not be called
	__device__ BxDFPtr get_bxdf(MaterialEvalInfo& info, void* localMem)
	{
		return nullptr;
	}
	__device__ SampledSpectrum Le(SampledWavelengths& swl)
	{
		RGBIlluminantSpectrum illum(*colorSpace, rgb);
		return illum.sample(swl);
	}
private:
	glm::vec3 rgb;
	RGBColorSpace* colorSpace;
};

constexpr uint32_t BxDFMaxSize = std::max({ sizeof(DiffuseBxDF), sizeof(DielectricBxDF), sizeof(ConductorBxDF)});