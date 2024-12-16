#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <thrust/random.h>
#include "taggedptr.h"
#include "utilities.h"
#include "spectrum.h"

#include "microfacet.h"

class DiffuseBxDF;
class DielectricBxDF;
class ConductorBxDF;
//class MetallicWorkflowBxDF;
//class BlinnPhongBxDF;
//class AsymConductorBxDF;
//class AsymDielectricBxDF;


class BxDFPtr: public TaggedPointer<DiffuseBxDF, DielectricBxDF, ConductorBxDF>
{
public:
    using TaggedPointer::TaggedPointer;
    __device__ SampledSpectrum sample_f(const glm::vec3& wo, glm::vec3& wi, float& pdf, thrust::default_random_engine& rng);
    __device__ SampledSpectrum eval(const glm::vec3& wo, const glm::vec3& wi, thrust::default_random_engine& rng);
    __device__ float pdf(const glm::vec3& wo, const glm::vec3& wi);
};


class DiffuseBxDF
{
public:
    __host__ __device__ DiffuseBxDF(const SampledSpectrum& ref) : reflectance(ref) {}
    __device__ SampledSpectrum sample_f(const glm::vec3& wo, glm::vec3& wi, float& pdf, thrust::default_random_engine& rng);
    __device__ SampledSpectrum eval(const glm::vec3& wo, const glm::vec3& wi, thrust::default_random_engine& rng);
    __device__ float pdf(const glm::vec3& wo, const glm::vec3& wi);
    SampledSpectrum reflectance;
};


class DielectricBxDF
{
public:
    __host__ __device__ DielectricBxDF(float eta) : eta(eta) {}
    __device__ SampledSpectrum sample_f(const glm::vec3& wo, glm::vec3& wi, float& pdf, thrust::default_random_engine& rng);
    __device__ SampledSpectrum eval(const glm::vec3& wo, const glm::vec3& wi, thrust::default_random_engine& rng);
    __device__ float pdf(const glm::vec3& wo, const glm::vec3& wi);
    float eta;
};

class ConductorBxDF
{
public:
    __device__ ConductorBxDF(const SampledSpectrum& eta, const SampledSpectrum& k, float alpha_x, float alpha_y) :eta(eta), k(k), dist(alpha_x, alpha_y) {}
    __device__ SampledSpectrum sample_f(const glm::vec3& wo, glm::vec3& wi, float& pdf, thrust::default_random_engine& rng);
    __device__ SampledSpectrum eval(const glm::vec3& wo, const glm::vec3& wi, thrust::default_random_engine& rng);
    __device__ float pdf(const glm::vec3& wo, const glm::vec3& wi);
    SampledSpectrum eta, k;
    TRDistribution dist;
};