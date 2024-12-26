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
	__device__ SampledSpectrum phi(SampledWavelengths lambda) const
	{
		//TODO
		assert(0);
		return {};
	}
	__device__ LightLiSample sample_Li(const LightSampleContext& ctx, glm::vec3 rand, SampledWavelengths lambda, bool allowIncompletePDF = false) const
	{
		Primitive& lightPrim = ctx.dev_primitives[m_primitiveID];
		Object& lightObj = ctx.dev_objects[lightPrim.objID];
		float prob = 1.0f;
		LightLiSample sample;
		glm::vec3 lightNormal;
		if (lightObj.type == GeomType::SPHERE)//Assume uniform scale of xyz
		{
			glm::vec3 originWorld = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(0.0), 1.0f));
			glm::vec3 rWorld = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(0.5, 0.0, 0.0), 0.0f));
			float R = glm::length(rWorld);
			glm::vec3 localSample = util_sample_hemisphere_uniform(glm::vec2(rand.x, rand.y));
			glm::vec3 N = glm::normalize(ctx.pi - originWorld);
			math::Frame frame = math::Frame::from_z(N);
			lightNormal = glm::normalize(frame.from_local(localSample));
			sample.pLight = originWorld + lightNormal * R;
			prob /= (TWO_PI * R * R);
		}
		else if (lightObj.type == GeomType::CUBE)//TODO: use quad light to replace all cubes
		{
			glm::vec3 v0 = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(-0.5f, -0.5f, -0.5f), 1.0f));
			glm::vec3 vx = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(1, 0, 0), 0.0f));
			glm::vec3 vy = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(0, 1, 0), 0.0f));
			glm::vec3 vz = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(0, 0, 1), 0.0f));

			float Axy = abs(glm::length(glm::cross(vx, vy)));
			float Axz = abs(glm::length(glm::cross(vx, vz)));
			float Ayz = abs(glm::length(glm::cross(vy, vz)));
			float area = 2 * (Axy + Axz + Ayz);
			prob /= area;
			float s = rand.x * area;
			double limit = Axy;
			float i = rand.y, j = rand.z;
			if (s < limit) {
				sample.pLight = v0 + vx * i + vy * j;
				lightNormal = glm::normalize(-vz);
			}
			else if (s < (limit += Axy)) {
				sample.pLight = v0 + vz + vx * i + vy * j;
				lightNormal = glm::normalize(vz);
			}
			else if (s < (limit += Axz)) {
				sample.pLight = v0 + vx * i + vz * j;
				lightNormal = glm::normalize(-vy);
			}
			else if (s < (limit += Axz)) {
				sample.pLight = v0 + vy + vx * i + vz * j;
				lightNormal = glm::normalize(vy);
			}
			else if (s < (limit += Ayz)) {
				sample.pLight = v0 + vy * i + vz * j;
				lightNormal = glm::normalize(-vx);
			}
			else {
				sample.pLight = v0 + vx + vy * i + vz * j;
				lightNormal = glm::normalize(vx);
			}
		}
		else //Triangle
		{
			glm::ivec3 tri = ctx.modelInfo.dev_triangles[lightObj.triangleStart + lightPrim.offset];
			glm::vec2 bary = util_math_uniform_sample_triangle(glm::vec2(rand.x, rand.y));
			const glm::vec3& v0 = ctx.modelInfo.dev_vertices[tri[0]];
			const glm::vec3& v1 = ctx.modelInfo.dev_vertices[tri[1]];
			const glm::vec3& v2 = ctx.modelInfo.dev_vertices[tri[2]];
			glm::vec3 v0w = multiplyMV(lightObj.Transform.transform, glm::vec4(v0, 1.0f));
			glm::vec3 v1w = multiplyMV(lightObj.Transform.transform, glm::vec4(v1, 1.0f));
			glm::vec3 v2w = multiplyMV(lightObj.Transform.transform, glm::vec4(v2, 1.0f));
			sample.pLight = v0w * bary[0] + v1w * bary[1] + v2w * (1 - bary[0] - bary[1]);
			glm::vec3 nNormal = glm::cross(v1w - v0w, v2w - v0w);
			float area = abs(glm::length(nNormal)) / 2;
			lightNormal = nNormal / (area > 0.0 ? area : 1e-8f);
			prob /= area;
		}
		glm::vec3 wl = sample.pLight - ctx.pi;
		sample.wi = glm::normalize(wl);
		float NoL = glm::dot(-sample.wi, glm::normalize(lightNormal));
		float rep_G = (glm::dot(wl, wl)) / NoL;
		sample.pdf = prob * rep_G;
		if (NoL < 0.0f)
		{
			if (!m_twoSided)
				sample.pdf = 0.0f;
			else
				sample.pdf = abs(sample.pdf);
		}
		SampledSpectrum Le = L(sample.pLight, lightNormal, glm::vec2(0.0), -wl, lambda);
		sample.L = Le;
		return sample;
	}
	__device__ float pdf_Li(const LightSampleContext& ctx, const glm::vec3& pLight, const glm::vec3& nLight, bool allowIncompletePDF = false)
	{
		float prob = 1.0f;
		Primitive& lightPrim = ctx.dev_primitives[m_primitiveID];
		Object& lightObj = ctx.dev_objects[lightPrim.objID];
		if (lightObj.type == GeomType::SPHERE)
		{
			glm::vec3 rWorld = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(0.5, 0.0, 0.0), 0.0f));
			float R = glm::length(rWorld);
			prob /= (TWO_PI * R * R);
		}
		else if (lightObj.type == GeomType::CUBE)
		{
			glm::vec3 vx = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(1, 0, 0), 0.0f));
			glm::vec3 vy = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(0, 1, 0), 0.0f));
			glm::vec3 vz = multiplyMV(lightObj.Transform.transform, glm::vec4(glm::vec3(0, 0, 1), 0.0f));
			float Axy = abs(glm::length(glm::cross(vx, vy)));
			float Axz = abs(glm::length(glm::cross(vx, vz)));
			float Ayz = abs(glm::length(glm::cross(vy, vz)));
			float area = 2 * (Axy + Axz + Ayz);
			prob /= area;
		}
		else
		{
			glm::ivec3 tri = ctx.modelInfo.dev_triangles[lightObj.triangleStart + lightPrim.offset];
			const glm::vec3& v0 = ctx.modelInfo.dev_vertices[tri[0]];
			const glm::vec3& v1 = ctx.modelInfo.dev_vertices[tri[1]];
			const glm::vec3& v2 = ctx.modelInfo.dev_vertices[tri[2]];
			glm::vec3 v0w = multiplyMV(lightObj.Transform.transform, glm::vec4(v0, 1.0f));
			glm::vec3 v1w = multiplyMV(lightObj.Transform.transform, glm::vec4(v1, 1.0f));
			glm::vec3 v2w = multiplyMV(lightObj.Transform.transform, glm::vec4(v2, 1.0f));
			glm::vec3 nNormal = glm::cross(v1w - v0w, v2w - v0w);
			float area = abs(glm::length(nNormal)) / 2;
			prob /= area;
		}
		glm::vec3 wl = ctx.pi - pLight;

		//TODO: handle double sided
		if (glm::dot(wl, nLight) < 0.0f)
			return 0.0f;
		return prob / (glm::dot(glm::normalize(wl), nLight) / (glm::dot(wl, wl)));
	}
	__device__ SampledSpectrum L(const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv, const glm::vec3& w, const SampledWavelengths& lambda) const
	{
		if (!m_twoSided && glm::dot(n, w) < 0)
			return SampledSpectrum(0.f);
		return m_scale * (*m_Lemit)(lambda);
	}
	__device__ SampledSpectrum Le(const Ray& ray, const SampledWavelengths& lambda) const
	{
		return {};
	}
private:
	int m_primitiveID;
	bool m_twoSided;
	const DenselySampledSpectrum* m_Lemit;
	float m_scale;
};