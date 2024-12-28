#include "lights.h"

GPU_FUNC SampledSpectrum LightPtr::phi(SampledWavelengths lambda) const
{
	auto f = [&](auto ptr)
		{
			return ptr->phi(lambda);
		};
	return Dispatch(f);
}

GPU_FUNC LightType LightPtr::type() const
{
	auto f = [&](auto ptr)
		{
			return ptr->type();
		};
	return Dispatch(f);
}

GPU_FUNC LightLiSample LightPtr::sample_Li(const LightSampleContext& ctx, glm::vec3 rand, SampledWavelengths lambda, bool allowIncompletePDF) const
{
	auto f = [&](auto ptr)
		{
			return ptr->sample_Li(ctx,rand,lambda,allowIncompletePDF);
		};
	return Dispatch(f);
}

GPU_FUNC float LightPtr::pdf_Li(const LightSampleContext& ctx, const glm::vec3& pLight, const glm::vec3& nLight, bool allowIncompletePDF) const
{
	auto f = [&](auto ptr)
		{
			return ptr->pdf_Li(ctx, pLight, nLight, allowIncompletePDF);
		};
	return Dispatch(f);
}

GPU_FUNC SampledSpectrum LightPtr::L(const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv, const glm::vec3& w, const SampledWavelengths& lambda) const
{
	auto f = [&](auto ptr)
		{
			return ptr->L(p,n,uv,w,lambda);
		};
	return Dispatch(f);
}

GPU_FUNC SampledSpectrum LightPtr::Le(const Ray& ray, const SampledWavelengths& lambda) const
{
	auto f = [&](auto ptr)
		{
			return ptr->Le(ray, lambda);
		};
	return Dispatch(f);
}

LightBase::SpectrumInstanceSet* LightBase::m_spectrumInstanceSet = nullptr;


GPU_FUNC LightLiSample DiffuseAreaLight::sample_Li(const LightSampleContext& ctx, glm::vec3 rand, SampledWavelengths lambda, bool allowIncompletePDF) const
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

GPU_FUNC float DiffuseAreaLight::pdf_Li(const LightSampleContext& ctx, const glm::vec3& pLight, const glm::vec3& nLight, bool allowIncompletePDF)  const
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

GPU_FUNC LightLiSample ImageInfiniteLight::sample_Li(const LightSampleContext& ctx, glm::vec3 rand, SampledWavelengths lambda, bool allowIncompletePDF) const
{
	LightLiSample sample;
	float mapPdf = 0.0f;
	glm::vec2 uv = m_dist->sample_continuous(glm::vec2(rand), &mapPdf);
	if (mapPdf == 0.0f)
	{
		return sample;
	}
	//TODO: transformation
	glm::vec3 wi = math::equirectangular_uv_to_dir(uv);
	float pdf = mapPdf / (math::pi * 4);
	sample.L = L({}, {}, {}, wi, lambda);
	sample.pdf = pdf;
	sample.wi = wi;
	//TODO: change this to a more proper value (according to scene bounds)
	sample.pLight = wi * 1e10f;
	return sample;
}

GPU_FUNC float ImageInfiniteLight::pdf_Li(const LightSampleContext& ctx, const glm::vec3& pLight, const glm::vec3& nLight, bool allowIncompletePDF) const
{
	//TODO: transformation
	glm::vec2 uv = math::equirectangular_dir_to_uv(glm::normalize(pLight));
	return m_dist->pdf(uv) / (4 * math::pi);
}

GPU_FUNC SampledSpectrum ImageInfiniteLight::L(const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv, const glm::vec3& w, const SampledWavelengths& lambda) const
{
	//TODO: transformation
	glm::vec2 mapUV = math::equirectangular_dir_to_uv(glm::normalize(w));
	float4 skyColorRGBA = tex2D<float4>(m_texture, mapUV.x, mapUV.y);
#if WHITE_FURNANCE_TEST
	glm::vec3 skyColor = glm::vec3(1.0, 1.0, 1.0);
#else
	glm::vec3 skyColor = glm::vec3(skyColorRGBA.x, skyColorRGBA.y, skyColorRGBA.z);
#endif
	skyColor *= m_scale;
	skyColor = glm::min(skyColor, m_max_radiance);
	const RGBColorSpace* colorSpace = RGBColorSpace_sRGB;
	RGBIlluminantSpectrum illumSpec(*colorSpace, skyColor);
	SampledSpectrum skyRadiance = illumSpec.sample(lambda);
	return skyRadiance;
}

