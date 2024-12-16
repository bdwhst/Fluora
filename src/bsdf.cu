#include "bsdf.h"
#include "mathUtils.h"
#include "spectrum.h"
#include "microfacet.h"
#include "randomUtils.h"
#include <cuda_runtime.h>

// Interface : Sample wi and return pdf & eval BSDF (cos wi premultiplied) 
__device__ SampledSpectrum BxDFPtr::sample_f(const glm::vec3& wo, glm::vec3& wi, float& pdf, thrust::default_random_engine& rng)
{
	auto op = [&](auto ptr) { return ptr->sample_f(wo, wi, pdf, rng); };
	return Dispatch(op);
}

// Interface : Eval BSDF
__device__ SampledSpectrum BxDFPtr::eval(const glm::vec3& wo, const glm::vec3& wi, thrust::default_random_engine& rng)
{
	auto op = [&](auto ptr) { return ptr->eval(wo, wi, rng); };
	return Dispatch(op);
}
// Interface : Eval PDF
__device__ float BxDFPtr::pdf(const glm::vec3& wo, const glm::vec3& wi)
{
	auto op = [&](auto ptr) { return ptr->pdf(wo, wi); };
	return Dispatch(op);
}

//__device__ inline glm::vec3 util_sample_hemisphere_uniform(const glm::vec2& random)
//{
//	float z = random.x;
//	float sq_1_z_2 = sqrt(max(1 - z * z, 0.0f));
//	float phi = TWO_PI * random.y;
//	return glm::vec3(cos(phi) * sq_1_z_2, sin(phi) * sq_1_z_2, z);
//}

//__device__ inline glm::vec2 util_sample_disk_uniform(const glm::vec2& random)
//{
//	float r = sqrt(random.x);
//	float theta = TWO_PI * random.y;
//	return glm::vec2(r * cos(theta), r * sin(theta));
//}
//
//__device__ inline glm::vec3 util_sample_hemisphere_cosine(const glm::vec2& random)
//{
//	glm::vec2 t = util_sample_disk_uniform(random);
//	return glm::vec3(t.x, t.y, sqrt(1 - t.x * t.x - t.y * t.y));
//}
//


//__device__ inline float util_math_tangent_space_clampedcos(const glm::vec3& w)
//{
//	return max(w.z, 0.0f);
//}

__device__ SampledSpectrum DiffuseBxDF::sample_f(const glm::vec3& wo, glm::vec3& wi, float& pdf, thrust::default_random_engine& rng)
{
	glm::vec2 random = util_sample2D(rng);
	//todo remove util_sample_hemisphere_cosine in intersection.h
    wi = math::sample_hemisphere_cosine(random);
	pdf = abs(math::cos_theta_vec(wi)) * INV_PI;
	float cosWi = abs(wi.z);
	SampledSpectrum f = reflectance * INV_PI * cosWi;
	return f;
}

__device__ SampledSpectrum DiffuseBxDF::eval(const glm::vec3& wo, const glm::vec3& wi, thrust::default_random_engine& rng)
{
	float cosWi = abs(wi.z);
    return reflectance * INV_PI * cosWi;
}

__device__ float DiffuseBxDF::pdf(const glm::vec3& wo, const glm::vec3& wi)
{
    return abs(math::cos_theta_vec(wi)) * INV_PI;
}

//__device__ inline float util_math_sin_cos_convert(float sinOrCos)
//{
//	return sqrt(max(1 - sinOrCos * sinOrCos, 0.0f));
//}

//__device__ inline float util_math_frensel_dielectric(float cosThetaI, float etaI, float etaT)
//{
//	cosThetaI = math::clamp(cosThetaI, -1.0f, 1.0f);
//	float sinThetaI = util_math_sin_cos_convert(cosThetaI);
//	float sinThetaT = etaI / etaT * sinThetaI;
//	if (sinThetaT >= 1) return 1;//total reflection
//	float cosThetaT = util_math_sin_cos_convert(sinThetaT);
//	float rparll = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
//	float rperpe = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
//	return (rparll * rparll + rperpe * rperpe) * 0.5;
//}
//
//__device__ inline bool util_geomerty_refract(const glm::vec3& wi, const glm::vec3& n, float eta, glm::vec3* wt)
//{
//	float cosThetaI = glm::dot(wi, n);
//	float sin2ThetaI = max(0.0f, 1 - cosThetaI * cosThetaI);
//	float sin2ThetaT = eta * eta * sin2ThetaI;
//	if (sin2ThetaT >= 1) return false;
//	float cosThetaT = sqrt(1 - sin2ThetaT);
//	*wt = eta * (-wi) + (eta * cosThetaI - cosThetaT) * n;
//	return true;
//}


__device__ SampledSpectrum DielectricBxDF::sample_f(const glm::vec3& wo, glm::vec3& wi, float& pdf, thrust::default_random_engine& rng)
{
	glm::vec2 random = util_sample2D(rng);
	//todo remove util_frensel_dielectric inside interaction.h
    float frensel = math::frensel_dielectric(abs(wo.z), 1.0f, eta);
	if (random.x < frensel)
	{
		wi = glm::vec3(-wo.x, -wo.y, wo.z);
		pdf = frensel;
		return SampledSpectrum(frensel);
	}
	else
	{
		glm::vec3 n = glm::dot(wo, glm::vec3(0, 0, 1)) > 0 ? glm::vec3(0, 0, 1) : glm::vec3(0, 0, -1);
		glm::vec3 refractedRay;
		if (!math::geomerty_refract(wo, n, 1 / eta, &refractedRay)) return SampledSpectrum(0.0f);
		wi = refractedRay;
		pdf = 1 - frensel;
		SampledSpectrum val((1 - frensel) / (eta * eta));
		return val;
	}
}

__device__ SampledSpectrum DielectricBxDF::eval(const glm::vec3& wo, const glm::vec3& wi, thrust::default_random_engine& rng)
{
    return SampledSpectrum(0.0f);
}

__device__ float DielectricBxDF::pdf(const glm::vec3& wo, const glm::vec3& wi)
{
    return 0.0f;
}

__device__ SampledSpectrum ConductorBxDF::sample_f(const glm::vec3& wo, glm::vec3& wi, float& pdf, thrust::default_random_engine& rng)
{
	//todo: handle effectly smooth bug
	using namespace math;
	if (dist.effectively_smooth())
	{
		wi = glm::vec3(-wo.x, -wo.y, wo.z);
		SampledSpectrum f = fr_complex(abs(cos_theta_vec(wi)), eta, k);
		pdf = 1.0f;
		return f;
	}
	else
	{
		glm::vec3 wm = dist.sample_wm(wo, util_sample2D(rng));
		wi = reflect(wo, wm);
		if (!sample_hemisphere(wo, wi))
		{
			pdf = 0.0f;
			return SampledSpectrum(0.0f);
		}
		float cosTheta_o = abs(cos_theta_vec(wo)), cosTheta_i = abs(cos_theta_vec(wi));
		if (cosTheta_o == 0 || cosTheta_i == 0) return SampledSpectrum(0.0f);
		pdf = dist.pdf(wo, wm) / (4.0f * abs_dot(wo, wm));
		SampledSpectrum F = fr_complex(abs_dot(wo, wm), eta, k);
		float D = dist.D(wm);
		float G = dist.G(wo, wi);
		SampledSpectrum f = D * F * G / (4 * cosTheta_o);//multipled by cos_wi
		return f;
	}
}



__device__ SampledSpectrum ConductorBxDF::eval(const glm::vec3& wo, const glm::vec3& wi, thrust::default_random_engine& rng)
{
	//todo: test
	using namespace math;
	glm::vec3 wm = wo + wi;
	if (wm.x * wm.x + wm.y * wm.y + wm.z * wm.z < 1e-9f) return SampledSpectrum(0.0f);
	wm = glm::normalize(wm);
	if ((wi.z > 0.0f && glm::dot(wm, wi) > 0.0f) && !dist.effectively_smooth())
	{
		SampledSpectrum F = fr_complex(abs_dot(wo, wm), eta, k);
		float cosTheta_o = abs(cos_theta_vec(wo));
		return dist.D(wm) * F * dist.G(wo, wi) / (4 * cosTheta_o);
	}
	else
	{
		return SampledSpectrum(0.0f);
	}
}

__device__ float ConductorBxDF::pdf(const glm::vec3& wo, const glm::vec3& wi)
{
	//todo: test
	using namespace math;
	glm::vec3 wm = wo + wi;
	if (wm.x * wm.x + wm.y * wm.y + wm.z * wm.z < 1e-9f) return 0.0f;
	wm = glm::normalize(wm);
	if ((wi.z > 0.0f && glm::dot(wm, wi) > 0.0f) && !dist.effectively_smooth())
	{
		return dist.pdf(wo, wm) / (4 * abs_dot(wo, wm));
	}
	return 0.0f;
}