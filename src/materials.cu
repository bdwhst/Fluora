#include "materials.h"

MaterialPtr MaterialPtr::create(const std::string& name, const BundledParams& params, Allocator alloc)
{
	if (name == "diffuse")
	{
		return DiffuseMaterial::create(params, alloc);
	}
	else if (name == "dielectric")
	{
		return DielectricMaterial::create(params, alloc);
	}
	else if (name == "conductor")
	{
		return ConductorMaterial::create(params, alloc);
	}
	else if (name == "emissive")
	{
		return EmissiveMaterial::create(params, alloc);
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