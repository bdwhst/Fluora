#pragma once

#include "taggedptr.h"
#include "spectrum.h"
#include "containers.h"

class HGPhaseFunction;
class PhaseFunctionPtr :public TaggedPointer<HGPhaseFunction>
{
public:
	__device__ float p(const glm::vec3& wo, const glm::vec3& wi) const;
	__device__ float sample_p(const glm::vec3& wo, const glm::vec2& u, glm::vec3* wi, float* pdf) const;
	__device__ float pdf(const glm::vec3& wo, const glm::vec3& wi) const;
};

class HGPhaseFunction
{
public:
	__device__ HGPhaseFunction(float g) :g(g) {}
	__device__ float p(const glm::vec3& wo, const glm::vec3& wi) const;
	__device__ float sample_p(const glm::vec3& wo, const glm::vec2& u, glm::vec3* wi, float* pdf) const;
	__device__ float pdf(const glm::vec3& wo, const glm::vec3& wi) const;
private:
	float g;
};

class HomogeneousMedium;
class GridMedium;
struct MediumProperties
{
	SampledSpectrum sigma_a, sigma_s;
	PhaseFunctionPtr phase;
	SampledSpectrum Le;
};

//struct RayMajorantSegment 
//{
//	float t_min, t_max;
//	SampledSpectrum sigma_maj;
//};

class HomogeneousMajorantIterator;
class DDAMajorantIterator;
class RGBGridMedium;
class NanoVDBMedium;

class RayMajorantIteratorPtr : public TaggedPointer<HomogeneousMajorantIterator, DDAMajorantIterator>
{
public:
	using TaggedPointer::TaggedPointer;
	__device__ bool next(float* t_min_p, float* t_max_p, SampledSpectrum* sigma_maj_p);
};

class MediumPtr : public TaggedPointer<HomogeneousMedium, GridMedium, RGBGridMedium, NanoVDBMedium>
{
public:
	using TaggedPointer::TaggedPointer;
	static MediumPtr create(const std::string& type, const BundledParams& params, const glm::mat4& world_from_medium, Allocator alloc);
	__device__ bool is_emissive() const;
	__device__ MediumProperties sample_point(const glm::vec3& point, const SampledWavelengths& lambda) const;
	//__device__ RayMajorantIteratorPtr sample_ray(const glm::vec3& ray_ori, const glm::vec3& ray_dir, float t_max, const SampledWavelengths& lambda, void* localMem) const;
};





