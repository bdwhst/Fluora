#include "materials.h"

MaterialHandle MaterialHandle::create(const std::string& name, const BundledParams& params, MaterialPool& pool)
{
	if (name == "diffuse")
	{
		return pool.add(DiffuseMaterial::create(params));
	}
	else if (name == "dielectric")
	{
		return pool.add(DielectricMaterial::create(params));
	}
	else if (name == "conductor")
	{
		return pool.add(ConductorMaterial::create(params));
	}
	else if (name == "emissive")
	{
		return pool.add(EmissiveMaterial::create(params));
	}
	else
	{
		std::cout << "Unsupported material: " << name << std::endl;
		exit(-1);
	}
}

__device__ BxDFPtr MaterialPtr::get_bxdf(MaterialEvalInfo& info, void* localMem)
{
	auto op = [&](auto ptr) { return ptr->get_bxdf(info, localMem); };
	return Dispatch(op);
}

__device__ glm::vec3 MaterialPtr::normal_mapping(const glm::vec2& uv)
{
	auto op = [&](auto ptr) { return ptr->normal_mapping(uv); };
	return Dispatch(op);
}

__device__ glm::vec3 MaterialBase::normal_mapping(const glm::vec2& uv)
{
	if (m_normalTexture)
	{
		float4 color = { 0,0,0,1 };
		color = tex2D<float4>(m_normalTexture, uv.x, uv.y);
		return glm::vec3(color.x, color.y, color.z);
	}
	return glm::vec3(0.0);
}

__device__ BxDFPtr DiffuseMaterial::get_bxdf(MaterialEvalInfo& info, void* localMem)
{
	DiffuseBxDF* bxdfPtr = (DiffuseBxDF*)localMem;
	glm::vec3 rgb = albedo;
	if (albedoMap)
	{
		float4 color = { 0,0,0,1 };
		color = tex2D<float4>(albedoMap, info.uv.x, info.uv.y);
		rgb.x = color.x;
		rgb.y = color.y;
		rgb.z = color.z;
	}
	RGBAlbedoSpectrum rgbSpec(*colorSpace, rgb);
	bxdfPtr->reflectance = rgbSpec.sample(info.swl);
	return bxdfPtr;
}