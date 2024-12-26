#include "misIntegrator.h"
#include "intersections.h"
#include "media.h"
__device__ cuda::atomic<int, cuda::thread_scope_device> numShadowRays{ 0 };


__global__ void sample_Ld_volume()
{

}

// Spectral & NEE MIS
// transport => f / pdf(current path)
// ru => pdf(lambda i of path sampling) / pdf(current path)
// rl => pdf(lambda i of light sampling) / pdf(current path)
__global__ void compute_intersection_bvh_volume_mis(
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


	bool scattered_in_medium = false, absorbed_in_medium = false, ternminated = false;


	if (ray.medium != -1)
	{
		thrust::default_random_engine& rng = pathSegment.rng;
		thrust::uniform_int_distribution<int> int_dist;
		thrust::default_random_engine tmaj_rng(int_dist(rng));

		float t_max = intersected_surface ? tmpIntersection.t : FLT_MAX;
		SampledSpectrum T_maj = sample_Tmaj(dev_sceneInfo.dev_media, ray, t_max, tmaj_rng, pathSegment.lambda, [&](const glm::vec3& p, MediumProperties mp, SampledSpectrum sigma_maj, SampledSpectrum T_maj) {
			if (!pathSegment.transport) {
				ternminated = true;
				return false;
			}
			// add medium emission here, so we don't add medium emission when absorption occurs
			if (mp.Le)
			{
				float pdf = sigma_maj[0] * T_maj[0];
				SampledSpectrum transport_p = pathSegment.transport * T_maj / pdf;
				SampledSpectrum r_e = pathSegment.r_u * sigma_maj * T_maj / pdf;
				if (r_e)
				{
					// Only spectral MIS
					SampledSpectrum Le = transport_p * mp.sigma_a * mp.Le / r_e.average();
					glm::vec3 sensorRGB = dev_sceneInfo.pixelSensor->to_sensor_rgb(Le, pathSegment.lambda);
					dev_film->add_radiance(sensorRGB, pathSegment.pixelIndex);
				}
			}
			float pAbsorb = mp.sigma_a[0] / sigma_maj[0];
			float pScatter = mp.sigma_s[0] / sigma_maj[0];
			float pNull = math::max(0.0f, 1 - pAbsorb - pScatter);
			thrust::uniform_real_distribution<float> u01(0, 1);
			float uMode = u01(tmaj_rng);
			if (uMode < pAbsorb)
			{
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

				float pdf = T_maj[0] * mp.sigma_s[0];
				pathSegment.transport *= T_maj * mp.sigma_s / pdf;
				pathSegment.r_u *= T_maj * mp.sigma_s / pdf;

				if (pathSegment.transport && pathSegment.r_u)
				{
					int currNumShadowRays = numShadowRays.fetch_add(1);
					// TODO: add shadow ray here

					glm::vec2 u(u01(tmaj_rng), u01(tmaj_rng));
					glm::vec3 wi;
					float pdf = 0.0f;
					float phase = mp.phase.sample_p(-ray.direction, u, &wi, &pdf);
					if (pdf == 0)
					{
						ternminated = true;
						return false;
					}
					ray.origin = p;
					ray.direction = wi;
					pathSegment.transport *= phase / pdf;
					// here there is no need to update r_u, since the phase function sampling is the same for all wavelength
					// r_l will be further multiplied by pdf of light sampling in scatter_on_intersection
					pathSegment.r_l = pathSegment.r_u / pdf;

					pathSegment.transport *= phase / pdf;
					pathSegment.prevSpecular = false;
					scattered_in_medium = true;
					return false;
				}
			}
			else
			{
				SampledSpectrum sigma_n = clamp_zero(sigma_maj - mp.sigma_a - mp.sigma_s);
				float pdf = T_maj[0] * sigma_n[0];
				pathSegment.transport *= T_maj * sigma_n / pdf;
				pathSegment.r_u *= T_maj * sigma_n / pdf;
				// ratio tracking is used for light sampling
				pathSegment.r_l *= T_maj * sigma_maj / pdf;
				return pathSegment.transport && pathSegment.r_u;
			}
			// Should not reach this
			assert(0);
			return false;
			});

		if (absorbed_in_medium || ternminated)
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

		// add remaining contribution of the ray inside the medium
		pathSegment.transport *= T_maj / T_maj[0];
		pathSegment.r_u *= T_maj / T_maj[0];
		pathSegment.r_l *= T_maj / T_maj[0];
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



__global__ void sample_Ld(int numRays, ShadowRaySegment* shadowRaySegments, LightSamplerPtr lightSampler, SceneInfoDev sceneInfo, RGBFilm* dev_film)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numRays) return;
	const ShadowRaySegment& shadowRaySegment = shadowRaySegments[idx];

	BxDFPtr bxdfPtr = BxDFPtr((void*)shadowRaySegment.bsdfData, shadowRaySegment.bsdfType);

	glm::vec3 pShading = shadowRaySegment.pWorld;
	thrust::uniform_real_distribution<float> u01(0, 1);
	thrust::default_random_engine& rng = shadowRaySegments[idx].rng;

	LightSampleContext ctx;
	ctx.pi = shadowRaySegment.pWorld;
	ctx.dev_primitives = sceneInfo.dev_primitives;
	ctx.dev_objects = sceneInfo.dev_objs;
	ctx.modelInfo = sceneInfo.modelInfo;

	SampledLight sampledLight = lightSampler.sample(ctx, u01(rng));
	if (sampledLight.pdf == 0.0f)
	{
		return;
	}
	LightPtr light = sampledLight.light;
	LightLiSample liSample = light.sample_Li(ctx, glm::vec3(u01(rng), u01(rng), u01(rng)), shadowRaySegment.lambda);

	float NoL = math::sgn(glm::dot(shadowRaySegment.normalWorld, liSample.wi));
	if (NoL < 0.0f && !(bxdfPtr.flags() & refraction))
	{
		return;
	}

	pShading += shadowRaySegment.normalWorld * NoL * SCATTER_ORIGIN_OFFSETMULT;

	if (liSample.pdf <= 0.0f || !liSample.L)
		return;

	if (!util_bvh_test_visibility(pShading, liSample.pLight, sceneInfo))
	{
		return;
	}

	math::Frame frame = math::Frame::from_z(shadowRaySegment.normalWorld);

	glm::vec3 wo = glm::normalize(frame.to_local(shadowRaySegment.woWorld));
	glm::vec3 wi = glm::normalize(frame.to_local(liSample.wi));
	SampledSpectrum f = bxdfPtr.eval(wo, wi, rng);

	if (!f)
	{
		return;
	}

	float p_l = sampledLight.pdf * liSample.pdf;
	float p_b = bxdfPtr.pdf(wo, wi);
	float w_l = math::sqr(p_l) / (math::sqr(p_l) + math::sqr(p_b));

	SampledSpectrum L = liSample.L * f * shadowRaySegment.transport * w_l / p_l;

	float threshold = 2.5f;
	if (L[0] > threshold || L[1] > threshold || L[2] > threshold || L[3] > threshold)
	{
		assert(1.0);
	}
	if (!L.is_nan() && !L.is_inf())
		dev_film->add_radiance(sceneInfo.pixelSensor->to_sensor_rgb(L, shadowRaySegment.lambda), shadowRaySegment.pixelIndex);
}

__device__ void gpu_memcpy(char* dst, char* src, const uint32_t size)
{
	GPU_UNROLL
		for (uint32_t i = 0; i < size; i++)
		{
			dst[i] = src[i];
		}
}

__global__ void scatter_on_intersection_mis(
	int iter
	, int depth
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, SceneInfoDev sceneInfo
	, int* rayValid
	, RGBFilm* dev_film
	, LightSamplerPtr lightSampler
	, ShadowRaySegment* shadowRaySegments
)
{
	extern __shared__ char sharedMemory[];
	char* bxdfBufferLocal = sharedMemory;

	MaterialPtr* materials = sceneInfo.dev_materials;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths) return;
	ShadeableIntersection intersection = shadeableIntersections[idx];
	const PathSegment& pathSegment = pathSegments[idx];
	thrust::default_random_engine& rng = pathSegments[idx].rng;
	thrust::uniform_real_distribution<float> u01(0, 1);
	MaterialPtr material = materials[intersection.materialId];

	if (material.Is<EmissiveMaterial>()) {
		SampledSpectrum Le = material.Cast<EmissiveMaterial>()->Le(pathSegment.lambda);
		SampledSpectrum L = pathSegment.transport * Le;
		rayValid[idx] = false;
		if (depth == 0 || pathSegment.prevSpecular)
		{
			dev_film->add_radiance(sceneInfo.pixelSensor->to_sensor_rgb(L, pathSegment.lambda), pathSegment.pixelIndex);
		}
		else if (lightSampler)
		{
			LightSampleContext ctx;
			ctx.pi = pathSegment.ray.origin;
			ctx.dev_primitives = sceneInfo.dev_primitives;
			ctx.dev_objects = sceneInfo.dev_objs;
			ctx.modelInfo = sceneInfo.modelInfo;
			LightPtr light = lightSampler.get_light(intersection.lightId);
			float p_l = lightSampler.pmf(ctx, light) * light.pdf_Li(ctx, intersection.worldPos, intersection.surfaceNormal);
			float w_b = math::sqr(pathSegment.lastMatPdf) / (math::sqr(pathSegment.lastMatPdf) + math::sqr(p_l));
			L *= w_b;
			dev_film->add_radiance(sceneInfo.pixelSensor->to_sensor_rgb(L, pathSegment.lambda), pathSegment.pixelIndex);
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
		bool isDeltaBsdf = bxdf.flags() & BxDFFlags::specular;
		thrust::uniform_int_distribution<int> int_dist;
		thrust::default_random_engine ld_rng(int_dist(rng));
		if (!isDeltaBsdf)
		{
			int currNumShadowRays = numShadowRays.fetch_add(1);
			shadowRaySegments[currNumShadowRays].transport = pathSegment.transport;
			shadowRaySegments[currNumShadowRays].lambda = pathSegment.lambda;
			shadowRaySegments[currNumShadowRays].normalWorld = glm::normalize(intersection.surfaceNormal);
			shadowRaySegments[currNumShadowRays].woWorld = -woInWorld;
			shadowRaySegments[currNumShadowRays].pWorld = intersection.worldPos;
			shadowRaySegments[currNumShadowRays].rng = ld_rng;
			shadowRaySegments[currNumShadowRays].bsdfType = bxdf.Tag();
			shadowRaySegments[currNumShadowRays].pixelIndex = pathSegment.pixelIndex;
			gpu_memcpy(shadowRaySegments[currNumShadowRays].bsdfData, bxdfBufferLocal + threadIdx.x * BxDFMaxSize, BxDFMaxSize);
			pathSegments[idx].prevSpecular = false;
		}
		else
		{
			pathSegments[idx].prevSpecular = true;
		}


		thrust::default_random_engine bxdf_rng(int_dist(rng));
		SampledSpectrum f = bxdf.sample_f(wo, wi, pdf, bxdf_rng);
		if (pdf > 0 && !pathSegments[idx].transport.is_nan() && !pathSegments[idx].transport.is_inf())
		{
			pathSegments[idx].transport *= f / pdf;
			glm::vec3 newDir = glm::normalize(frame.from_local(wi));
			glm::vec3 offset = glm::dot(newDir, N) < 0 ? -N : N;
			float offsetMult = !material.Is<DielectricMaterial>() ? SCATTER_ORIGIN_OFFSETMULT : SCATTER_ORIGIN_OFFSETMULT * 100.0f;
			pathSegments[idx].ray.origin = intersection.worldPos + offset * offsetMult;
			pathSegments[idx].ray.direction = newDir;
			pathSegments[idx].lastMatPdf = pdf;
			rayValid[idx] = true;
		}
		else
		{
			rayValid[idx] = false;
		}

	}
}

//__global__ void scatter_on_intersection_mis(
//	int iter
//	, int num_paths
//	, ShadeableIntersection* shadeableIntersections
//	, PathSegment* pathSegments
//	, SceneInfoDev sceneInfo
//	, int* rayValid
//	, glm::vec3* image
//)
//{
//	int idx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (idx >= num_paths) return;
//	ShadeableIntersection intersection = shadeableIntersections[idx];
//	thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
//	thrust::uniform_real_distribution<float> u01(0, 1);
//
//	Material* materials = sceneInfo.dev_materials;
//	Material material = materials[intersection.materialId];
//	glm::vec3 materialColor = material.color;
//#if VIS_NORMAL
//	image[pathSegments[idx].pixelIndex] += (glm::normalize(intersection.surfaceNormal));
//	rayValid[idx] = 0;
//	return;
//#endif
//
//	// If the material indicates that the object was a light, "light" the ray
//	if (material.type == MaterialType::emitting) {
//		int lightPrimId = intersection.primitiveId;
//		
//		float matPdf = pathSegments[idx].lastMatPdf;
//		if (matPdf > 0.0)
//		{
//			float G = util_math_solid_angle_to_area(intersection.worldPos, intersection.surfaceNormal, pathSegments[idx].ray.origin);
//			//We do not know the value of light pdf(of last intersection point) of the sample taken from bsdf sampling unless we hit a light
//			float lightPdf = lights_sample_pdf(sceneInfo, lightPrimId);
//			//Computing weights from last intersection point
//			float misW = util_mis_weight(matPdf * G, lightPdf);
//			pathSegments[idx].transport *= (materialColor * material.emittance * misW);
//		}
//		else//This ray hits a light directly
//		{
//			pathSegments[idx].transport *= (materialColor * material.emittance);
//		}
//		rayValid[idx] = 0;
//		if (!util_math_is_nan(pathSegments[idx].transport))
//			image[pathSegments[idx].pixelIndex] += pathSegments[idx].transport;
//	}
//	else {
//		//Prepare normal and wo for sample
//		glm::vec3& woInWorld = pathSegments[idx].ray.direction;
//		glm::vec3 nMap = glm::vec3(0, 0, 1);
//		if (material.normalMap != 0)
//		{
//			float4 nMapCol = tex2D<float4>(material.normalMap, intersection.uv.x, intersection.uv.y);
//			nMap.x = nMapCol.x;
//			nMap.y = nMapCol.y;
//			nMap.z = nMapCol.z;
//			nMap = glm::pow(nMap, glm::vec3(1 / 2.2f));
//			nMap = nMap * 2.0f - 1.0f;
//			nMap = glm::normalize(nMap);
//		}
//		glm::vec3 N = glm::normalize(intersection.surfaceNormal);
//		glm::vec3 B, T;
//		if (material.normalMap != 0)
//		{
//			T = intersection.surfaceTangent;
//			T = glm::normalize(T - N * glm::dot(N, T));
//			B = glm::cross(N, T);
//			N = glm::normalize(T * nMap.x + B * nMap.y + N * nMap.z);
//		}
//		else
//		{
//			util_math_get_TBN_pixar(N, &T, &B);
//		}
//		glm::mat3 TBN(T, B, N);
//		glm::vec3 wo = glm::transpose(TBN) * (-woInWorld);
//		wo = glm::normalize(wo);
//		float pdf = 0;
//		glm::vec3 wi, bxdf;
//		glm::vec3 random = glm::vec3(u01(rng), u01(rng), u01(rng));
//		float cosWi = 0;
//		if (material.type == MaterialType::frenselSpecular)
//		{
//			glm::vec2 iors = glm::dot(woInWorld, N) < 0 ? glm::vec2(1.0, material.indexOfRefraction) : glm::vec2(material.indexOfRefraction, 1.0);
//			bxdf = bxdf_frensel_specular_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor, materialColor, iors);
//			cosWi = 1.0;
//		}
//		else
//		{
//			float roughness = material.roughness, metallic = material.metallic;
//			float specExp = material.specExponent;
//			float4 color = { 0,0,0,1 };
//			float alpha = 1.0f;
//			//Texture mapping
//			if (material.baseColorMap != 0)
//			{
//				color = tex2D<float4>(material.baseColorMap, intersection.uv.x, intersection.uv.y);
//				materialColor.x = color.x;
//				materialColor.y = color.y;
//				materialColor.z = color.z;
//				alpha = color.w;
//			}
//			if (material.metallicRoughnessMap != 0)
//			{
//				color = tex2D<float4>(material.metallicRoughnessMap, intersection.uv.x, intersection.uv.y);
//				roughness = color.y;
//				metallic = color.z;
//			}
//			//Sampling lights
//			glm::vec3 lightPos, lightNormal, emissive = glm::vec3(0);
//			float light_pdf = -1.0;
//			glm::vec3 offseted_pos = intersection.worldPos + N * SCATTER_ORIGIN_OFFSETMULT;
//			lights_sample(sceneInfo, glm::vec3(u01(rng), u01(rng), u01(rng)), offseted_pos, N, &lightPos, &lightNormal, &emissive, &light_pdf);
//			glm::vec3 light_bxdf = glm::vec3(0);
//			
//			if (emissive.x > 0.0 || emissive.y > 0.0 || emissive.z > 0.0)
//			{
//				glm::vec3 light_wi = lightPos - offseted_pos;
//				light_wi = glm::normalize(glm::transpose(TBN) * (light_wi));
//				float G = util_math_solid_angle_to_area(lightPos, lightNormal, offseted_pos);
//				float mat_pdf = -1.0f;
//				if (material.type == MaterialType::metallicWorkflow)
//				{
//					mat_pdf = bxdf_metallic_workflow_pdf(wo, light_wi, materialColor, metallic, roughness);
//					light_bxdf = bxdf_metallic_workflow_eval(wo, light_wi, materialColor, metallic, roughness);
//				}
//				else if (material.type == MaterialType::microfacet)
//				{
//					mat_pdf = bxdf_microfacet_pdf(wo, light_wi, roughness);
//					light_bxdf = bxdf_microfacet_eval(wo, light_wi, materialColor, roughness);
//				}
//				else if (material.type == MaterialType::blinnphong)
//				{
//					mat_pdf = bxdf_blinn_phong_pdf(wo, light_wi, specExp);
//					light_bxdf = bxdf_blinn_phong_eval(wo, light_wi, materialColor, specExp);
//				}
//				else
//				{
//					mat_pdf = bxdf_diffuse_pdf(wo, light_wi);
//					light_bxdf = bxdf_diffuse_eval(wo, light_wi, materialColor);
//				}
//				float misW = util_mis_weight(light_pdf, mat_pdf * G);
//				image[pathSegments[idx].pixelIndex] += pathSegments[idx].transport * light_bxdf * util_math_tangent_space_clampedcos(light_wi) * emissive * misW * G / light_pdf;
//			}
//			//Sampling material bsdf
//			if (material.type == MaterialType::metallicWorkflow)
//			{	
//				bxdf = bxdf_metallic_workflow_sample_f(wo, &wi, random, &pdf, materialColor, metallic, roughness);
//			}
//			else if (material.type == MaterialType::microfacet)
//			{
//				bxdf = bxdf_microfacet_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor, roughness);
//			}
//			else if (material.type == MaterialType::blinnphong)
//			{
//				bxdf = bxdf_blinn_phong_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor, specExp);
//			}
//			else//diffuse
//			{
//				if (alpha <= ALPHA_CUTOFF)
//				{
//					bxdf = pathSegments[idx].remainingBounces == 0 ? glm::vec3(0, 0, 0) : glm::vec3(1, 1, 1);
//					wi = -wo;
//					pdf = util_math_tangent_space_clampedcos(wi);
//				}
//				else
//				{
//					bxdf = bxdf_diffuse_sample_f(wo, &wi, glm::vec2(random.x, random.y), &pdf, materialColor);
//				}
//
//			}
//			cosWi = util_math_tangent_space_clampedcos(wi);
//		}
//		if (pdf > 0)
//		{
//			pathSegments[idx].transport *= bxdf * cosWi / pdf;
//			glm::vec3 newDir = glm::normalize(TBN * wi);
//			glm::vec3 offset = glm::dot(newDir, N) < 0 ? -N : N;
//			float offsetMult = material.type != MaterialType::frenselSpecular ? SCATTER_ORIGIN_OFFSETMULT : SCATTER_ORIGIN_OFFSETMULT * 100.0f;
//			pathSegments[idx].ray.origin = intersection.worldPos + offset * offsetMult;
//			pathSegments[idx].ray.direction = newDir;
//			pathSegments[idx].lastMatPdf = pdf;
//			rayValid[idx] = 1;
//		}
//		else
//		{
//			rayValid[idx] = 0;
//		}
//
//	}
//}



__global__ void scatter_on_intersection_volume_mis(
	int iter
	, int depth
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, SceneInfoDev sceneInfo
	, int* rayValid
	, RGBFilm* dev_film
	, LightSamplerPtr lightSampler
)
{
	extern __shared__ char sharedMemory[];
	char* bxdfBufferLocal = sharedMemory;

	MaterialPtr* materials = sceneInfo.dev_materials;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_paths) return;
	ShadeableIntersection intersection = shadeableIntersections[idx];
	// Set up the RNG
	// LOOK: this is how you use thrust's RNG! Please look at
	// makeSeededRandomEngine as well.
	const PathSegment& pathSegment = pathSegments[idx];
	thrust::default_random_engine& rng = pathSegments[idx].rng;
	thrust::uniform_real_distribution<float> u01(0, 1);
	// scattered in media
	if (intersection.materialId == -1)
	{
		rayValid[idx] = true;
		return;
	}
	MaterialPtr material = materials[intersection.materialId];

	// If the material indicates that the object was a light, "light" the ray
	if (material.Is<EmissiveMaterial>()) {
		SampledSpectrum L(0.0f);
		if (depth == 0 || pathSegment.prevSpecular)
		{
			L = pathSegment.transport * material.Cast<EmissiveMaterial>()->Le(pathSegment.lambda) / pathSegment.r_u.average();
		}
		else if (lightSampler)
		{
			int lightId = intersection.lightId;
			LightSampleContext ctx;
			ctx.pi = pathSegment.ray.origin;
			ctx.dev_primitives = sceneInfo.dev_primitives;
			ctx.dev_objects = sceneInfo.dev_objs;
			ctx.modelInfo = sceneInfo.modelInfo;
			LightPtr light = lightSampler.get_light(lightId);
			float lightPdf = lightSampler.pmf(ctx, light) * light.pdf_Li(ctx, intersection.worldPos, intersection.surfaceNormal);
			SampledSpectrum r_l = pathSegment.r_l * lightPdf;
			L = pathSegment.transport * material.Cast<EmissiveMaterial>()->Le(pathSegment.lambda) / (pathSegment.r_u + r_l).average();
		}

		rayValid[idx] = false;
		if (!L.is_nan())
		{
			dev_film->add_radiance(sceneInfo.pixelSensor->to_sensor_rgb(L, pathSegments[idx].lambda), pathSegments[idx].pixelIndex);
		}
	}
	else {
		glm::vec3& woInWorld = pathSegments[idx].ray.direction;
		glm::vec3 nMap = glm::vec3(0, 0, 1);

		glm::vec3 N = glm::normalize(intersection.surfaceNormal);
		glm::vec3 B, T;

		math::Frame frame = math::Frame::from_z(N);
		glm::vec3 wo = frame.to_local(-woInWorld);
		wo = glm::normalize(wo);
		float pdf = 0;
		glm::vec3 wi;

		MaterialEvalInfo info(wo, intersection.uv, pathSegments[idx].lambda);

		BxDFPtr bxdf = material.get_bxdf(info, bxdfBufferLocal + threadIdx.x * BxDFMaxSize);

		bool isDelta = bxdf.flags() & BxDFFlags::specular;
		if (!isDelta)
		{
			int currNumShadowRays = numShadowRays.fetch_add(1);
			// TODO: add shadow ray here

		}

		thrust::uniform_int_distribution<int> int_dist;
		thrust::default_random_engine bxdf_rng(int_dist(rng));
		SampledSpectrum f = bxdf.sample_f(wo, wi, pdf, bxdf_rng);

		if (pdf > 0)
		{
			pathSegments[idx].transport *= f / pdf;
			pathSegments[idx].r_l = pathSegments[idx].r_u / pdf;
			pathSegments[idx].prevSpecular = isDelta;
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
