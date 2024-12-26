#include "naiveIntegrator.h"
#include "randomUtils.h"
#include "intersections.h"
#include "media.h"

__global__ void compute_intersection_bvh_no_volume(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, SceneInfoDev dev_sceneInfo
	, ShadeableIntersection* intersections
	, int* rayValid
	, RGBFilm* dev_film
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index >= num_paths) return;
	PathSegment& pathSegment = pathSegments[path_index];
	Ray& ray = pathSegment.ray;
	glm::vec3 rayDir = pathSegment.ray.direction;
	glm::vec3 rayOri = pathSegment.ray.origin;
	float x = fabs(rayDir.x), y = fabs(rayDir.y), z = fabs(rayDir.z);
	int axis = x > y && x > z ? 0 : (y > z ? 1 : 2);
	int sgn = rayDir[axis] > 0 ? 0 : 1;
	int d = (axis << 1) + sgn;
	const MTBVHGPUNode* currArray = dev_sceneInfo.dev_mtbvhArray + d * dev_sceneInfo.bvhDataSize;
	int curr = 0;
	ShadeableIntersection tmpIntersection;
	tmpIntersection.t = FLT_MAX;
	bool intersected = false;
	while (curr >= 0 && curr < dev_sceneInfo.bvhDataSize)
	{
		bool outside = true;
		float boxt = boundingBoxIntersectionTest(currArray[curr].bbox, ray, outside);
		if (!outside) boxt = EPSILON;
		if (boxt > 0 && boxt < tmpIntersection.t)
		{
			if (currArray[curr].startPrim != -1)//leaf node
			{
				int start = currArray[curr].startPrim, end = currArray[curr].endPrim;
				bool intersect = util_bvh_leaf_intersect(start, end, dev_sceneInfo, &ray, &tmpIntersection);
				intersected = intersected || intersect;
			}
			curr = currArray[curr].hitLink;
		}
		else
		{
			curr = currArray[curr].missLink;
		}
	}

	rayValid[path_index] = intersected;
	if (intersected)
	{
		intersections[path_index] = tmpIntersection;
		pathSegment.remainingBounces--;
	}
	else if (dev_sceneInfo.skyboxObj)
	{
		glm::vec2 uv = util_sample_spherical_map(glm::normalize(rayDir));
		float4 skyColorRGBA = tex2D<float4>(dev_sceneInfo.skyboxObj, uv.x, uv.y);
#if WHITE_FURNANCE_TEST
		glm::vec3 skyColor = glm::vec3(1.0, 1.0, 1.0);
#else
		glm::vec3 skyColor = glm::vec3(skyColorRGBA.x, skyColorRGBA.y, skyColorRGBA.z);
#endif
		const RGBColorSpace* colorSpace = RGBColorSpace_sRGB;
		RGBIlluminantSpectrum illumSpec(*colorSpace, skyColor);
		SampledSpectrum skyRadiance = illumSpec.sample(pathSegment.lambda);
		glm::vec3 sensorRGB = dev_sceneInfo.pixelSensor->to_sensor_rgb(pathSegment.transport * skyRadiance, pathSegment.lambda);
		dev_film->add_radiance(sensorRGB, pathSegment.pixelIndex);
	}
}


// Does not handle surface intersection
__global__ void compute_intersection_bvh_volume_naive(
	int iter
	, int depth
	, int num_paths
	, PathSegment* pathSegments
	, SceneInfoDev dev_sceneInfo
	, ShadeableIntersection* intersections
	, int* rayValid
	, RGBFilm* dev_film
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index >= num_paths) return;
	PathSegment& pathSegment = pathSegments[path_index];
	Ray& ray = pathSegment.ray;
	glm::vec3 rayDir = pathSegment.ray.direction;
	glm::vec3 rayOri = pathSegment.ray.origin;
	float x = fabs(rayDir.x), y = fabs(rayDir.y), z = fabs(rayDir.z);
	int axis = x > y && x > z ? 0 : (y > z ? 1 : 2);
	int sgn = rayDir[axis] > 0 ? 0 : 1;
	int d = (axis << 1) + sgn;
	const MTBVHGPUNode* currArray = dev_sceneInfo.dev_mtbvhArray + d * dev_sceneInfo.bvhDataSize;
	int curr = 0;
	ShadeableIntersection tmpIntersection;
	tmpIntersection.t = FLT_MAX;
	tmpIntersection.materialId = -1;
	bool intersected_surface = false;
	while (curr >= 0 && curr < dev_sceneInfo.bvhDataSize)
	{
		bool outside = true;
		float boxt = boundingBoxIntersectionTest(currArray[curr].bbox, ray, outside);
		if (!outside) boxt = EPSILON;
		if (boxt > 0 && boxt < tmpIntersection.t)
		{
			if (currArray[curr].startPrim != -1)//leaf node
			{
				int start = currArray[curr].startPrim, end = currArray[curr].endPrim;
				bool intersect = util_bvh_leaf_intersect(start, end, dev_sceneInfo, &ray, &tmpIntersection);
				intersected_surface = intersected_surface || intersect;
			}
			curr = currArray[curr].hitLink;
		}
		else
		{
			curr = currArray[curr].missLink;
		}
	}
	pathSegment.lambda.terminate_secondary();
	bool scattered_in_medium = false, absorbed_in_medium = false;

	// FIXED: Triangle intersection is now watertight
	if (intersected_surface && depth == 0 && ray.medium != -1)
	{
		assert(0);
	}

	if (ray.medium != -1)
	{
		thrust::default_random_engine& rng = pathSegment.rng;
		thrust::uniform_int_distribution<int> int_dist;
		thrust::default_random_engine tmaj_rng(int_dist(rng));

		float t_max = intersected_surface ? tmpIntersection.t : FLT_MAX;
		sample_Tmaj(dev_sceneInfo.dev_media, ray, t_max, tmaj_rng, pathSegment.lambda, [&](const glm::vec3& p, MediumProperties mp, SampledSpectrum sigma_maj, SampledSpectrum Tmaj) {
			float pAbsorb = mp.sigma_a[0] / sigma_maj[0];
			float pScatter = mp.sigma_s[0] / sigma_maj[0];
			float pNull = math::max(0.0f, 1 - pAbsorb - pScatter);
			if (pNull == 1.0f)
			{
				return true;
			}
			thrust::uniform_real_distribution<float> u01(0, 1);
			float uMode = u01(tmaj_rng);
			if (uMode < pAbsorb)
			{
				glm::vec3 sensorRGB = dev_sceneInfo.pixelSensor->to_sensor_rgb(pathSegment.transport * mp.Le, pathSegment.lambda);
				dev_film->add_radiance(sensorRGB, pathSegment.pixelIndex);
				absorbed_in_medium = true;
				return false;
			}
			else if (uMode >= pAbsorb && uMode < pAbsorb + pScatter)
			{
				pathSegment.remainingBounces--;
				int bounces = pathSegment.remainingBounces;
				if (bounces == 0)
				{
					return false;
				}

				glm::vec2 u(u01(tmaj_rng), u01(tmaj_rng));
				glm::vec3 wi;
				float pdf = 0.0f;
				float phase = mp.phase.sample_p(-ray.direction, u, &wi, &pdf);
				if (pdf == 0)
				{
					return false;
				}
				ray.origin = p;
				ray.direction = wi;

				pathSegment.transport *= phase / pdf;
				scattered_in_medium = true;
				return false;
			}
			else
			{
				return true;
			}
			});
	}
	if (absorbed_in_medium)
	{
		rayValid[path_index] = false;
		return;
	}

	// If real scatter occurs, mark materialId as -1
	if (scattered_in_medium)
	{
		intersections[path_index].materialId = -1;
		rayValid[path_index] = true;
		return;
	}
	// If there is no real scatter and a intersection with surface occurs
	// We are intersecting with a medium interface or a light surface
	// Continue travese through the current ray dir, but change the origin to be the intersection point
	if (intersected_surface)
	{
		intersections[path_index] = tmpIntersection;
		ray.origin = tmpIntersection.worldPos + ray.direction * SCATTER_ORIGIN_OFFSETMULT;
		rayValid[path_index] = true;
		return;
	}
	// If there is no scatter in media and intersection with surface
	// Try to read the radiance from skybox
	if (dev_sceneInfo.skyboxObj)
	{
		glm::vec2 uv = util_sample_spherical_map(glm::normalize(rayDir));
		float4 skyColorRGBA = tex2D<float4>(dev_sceneInfo.skyboxObj, uv.x, uv.y);
#if WHITE_FURNANCE_TEST
		glm::vec3 skyColor = glm::vec3(1.0, 1.0, 1.0);
#else
		glm::vec3 skyColor = glm::vec3(skyColorRGBA.x, skyColorRGBA.y, skyColorRGBA.z);
#endif
		const RGBColorSpace* colorSpace = RGBColorSpace_sRGB;
		RGBIlluminantSpectrum illumSpec(*colorSpace, skyColor);
		SampledSpectrum skyRadiance = illumSpec.sample(pathSegment.lambda);
		glm::vec3 sensorRGB = dev_sceneInfo.pixelSensor->to_sensor_rgb(pathSegment.transport * skyRadiance, pathSegment.lambda);
		dev_film->add_radiance(sensorRGB, pathSegment.pixelIndex);
		rayValid[path_index] = false;
	}
}


__global__ void scatter_on_intersection(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, SceneInfoDev sceneInfo
	, int* rayValid
	, RGBFilm* dev_film
)
{
	extern __shared__ char sharedMemory[];
	char* bxdfBufferLocal = sharedMemory;

	MaterialPtr* materials = sceneInfo.dev_materials;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths) return;
	ShadeableIntersection intersection = shadeableIntersections[idx];
	thrust::default_random_engine& rng = pathSegments[idx].rng;
	thrust::uniform_real_distribution<float> u01(0, 1);
	// scattered in media
	if (intersection.materialId == -1)
	{
		rayValid[idx] = true;
		return;
	}
	MaterialPtr material = materials[intersection.materialId];

	if (material.Is<EmissiveMaterial>()) {
		SampledSpectrum Le = material.Cast<EmissiveMaterial>()->Le(pathSegments[idx].lambda);
		SampledSpectrum L = pathSegments[idx].transport * Le;
		rayValid[idx] = false;
		if (!L.is_nan())
		{
			dev_film->add_radiance(sceneInfo.pixelSensor->to_sensor_rgb(L, pathSegments[idx].lambda), pathSegments[idx].pixelIndex);
		}
	}
	else {
		// For now if we encounter some non-emissive surface while rendering volumetrics, just error exit
		assert(sceneInfo.containsVolume == false);
		glm::vec3& woInWorld = pathSegments[idx].ray.direction;
		glm::vec3 N = glm::normalize(intersection.surfaceNormal);
		math::Frame frame = math::Frame::from_z(N);
		glm::vec3 wo = frame.to_local(-woInWorld);
		wo = glm::normalize(wo);
		float pdf = 0;
		glm::vec3 wi;

		MaterialEvalInfo info(wo, intersection.uv, pathSegments[idx].lambda);

		BxDFPtr bxdf = material.get_bxdf(info, bxdfBufferLocal + threadIdx.x * BxDFMaxSize);

		thrust::uniform_int_distribution<int> int_dist;
		thrust::default_random_engine ld_rng(int_dist(rng));
		thrust::default_random_engine bxdf_rng(int_dist(rng));
		SampledSpectrum f = bxdf.sample_f(wo, wi, pdf, bxdf_rng);

		//glm::vec3 wi, bxdf;
		//glm::vec3 random = glm::vec3(u01(rng), u01(rng), u01(rng));
		//float cosWi = 0;
		//if (material.type == MaterialType::metallicWorkflow)
		//{
		//	float4 color = { 0,0,0,1 };
		//	float roughness = material.roughness, metallic = material.metallic;
		//	if (material.baseColorMap != 0)
		//	{
		//		color = tex2D<float4>(material.baseColorMap, intersection.uv.x, intersection.uv.y);
		//		materialColor.x = color.x;
		//		materialColor.y = color.y;
		//		materialColor.z = color.z;
		//	}
		//	if (material.metallicRoughnessMap != 0)
		//	{
		//		color = tex2D<float4>(material.metallicRoughnessMap, intersection.uv.x, intersection.uv.y);
		//		roughness = color.y;
		//		metallic = color.z;
		//	}

		//	bxdf = bxdf_metallic_workflow_sample_f(wo, &wi, random, &pdf, materialColor, metallic, roughness);
		//	cosWi = util_math_tangent_space_clampedcos(wi);
		//}
		//else if (material.type == MaterialType::frenselSpecular)
		//{
		//	glm::vec2 iors = glm::dot(woInWorld, N) < 0 ? glm::vec2(1.0, material.indexOfRefraction) : glm::vec2(material.indexOfRefraction, 1.0);
		//	bxdf = bxdf_frensel_specular_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor, materialColor, iors);
		//	cosWi = 1.0;
		//}
		//else if (material.type == MaterialType::microfacet)
		//{
		//	bxdf = bxdf_microfacet_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor, material.roughness);
		//	cosWi = util_math_tangent_space_clampedcos(wi);
		//}
		//else if (material.type == MaterialType::blinnphong)
		//{
		//	bxdf = bxdf_blinn_phong_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor, material.specExponent);
		//	cosWi = util_math_tangent_space_clampedcos(wi);
		//}
		//else if (material.type == MaterialType::asymMicrofacet)
		//{
		//	if(material.asymmicrofacet.type == conductor)
		//		bxdf = bxdf_asymConductor_sample_f(wo, &wi, rng, &pdf, material.asymmicrofacet, NUM_MULTI_SCATTER_BOUNCE);
		//	else
		//		bxdf = bxdf_asymDielectric_sample_f(wo, &wi, rng, &pdf, material.asymmicrofacet, NUM_MULTI_SCATTER_BOUNCE);
		//	cosWi = 1.0f;
		//}
		//else//diffuse
		//{
		//	float4 color = { 0,0,0,1 };
		//	if (material.baseColorMap != 0)
		//	{
		//		color = tex2D<float4>(material.baseColorMap, intersection.uv.x, intersection.uv.y);
		//		materialColor.x = color.x;
		//		materialColor.y = color.y;
		//		materialColor.z = color.z;
		//	}
		//	
		//	if (color.w <= ALPHA_CUTOFF)
		//	{
		//		bxdf = pathSegments[idx].remainingBounces == 0 ? glm::vec3(0, 0, 0) : glm::vec3(1, 1, 1);
		//		wi = -wo;
		//		pdf = abs(wi.z);
		//	}
		//	else
		//	{
		//		bxdf = bxdf_diffuse_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor);
		//	}
		//	cosWi = abs(wi.z);

		//}
		if (pdf > 0 && !pathSegments[idx].transport.is_nan() && !pathSegments[idx].transport.is_inf())
		{
			pathSegments[idx].transport *= f / pdf;
			glm::vec3 newDir = glm::normalize(frame.from_local(wi));
			glm::vec3 offset = glm::dot(newDir, N) < 0 ? -N : N;
			float offsetMult = !material.Is<DielectricMaterial>() ? SCATTER_ORIGIN_OFFSETMULT : SCATTER_ORIGIN_OFFSETMULT * 100.0f;
			pathSegments[idx].ray.origin = intersection.worldPos + offset * offsetMult;
			pathSegments[idx].ray.direction = newDir;
			rayValid[idx] = true;
		}
		else
		{
			rayValid[idx] = false;
		}

	}
}
