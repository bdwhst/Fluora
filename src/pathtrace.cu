#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "randomUtils.h"
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "media.h"
//#include "materials.h"







__device__ inline bool util_math_is_nan(const glm::vec3& v)
{
	return (v.x != v.x) || (v.y != v.y) || (v.z != v.z);
}



//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, RGBFilm* dev_film) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = dev_film->get_image()[index];

		glm::vec3 color;
#if TONEMAPPING
		color = pix / (float)iter;
		color = max(color, glm::vec3(0.0f));
		color = util_postprocess_ACESFilm(color);
		color = color * 255.0f;
#else
		color = pix / (float)iter;
		float r = color.r, g = color.g, b = color.b;
		color = glm::clamp(glm::vec3(r, g, b) * 255.0f, glm::vec3(0.0f), glm::vec3(255.0f));

#endif
		if (util_math_is_nan(pix))
		{
			pbo[index].x = 255;
			pbo[index].y = 192;
			pbo[index].z = 203;
		}
		else
		{
			// Each thread writes one pixel location in the texture (textel)
			pbo[index].x = color.x;
			pbo[index].y = color.y;
			pbo[index].z = color.z;
		}
		pbo[index].w = 0;
	}
}


static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Object* dev_objs = NULL;
static MaterialPtr* dev_materials = NULL;
static MediumPtr* dev_media = NULL;
static MTBVHGPUNode* dev_mtbvhArray = NULL;
static Primitive* dev_primitives = NULL;
static glm::ivec3* dev_triangles = NULL;
static glm::vec3* dev_vertices = NULL;
static glm::vec2* dev_uvs = NULL;
static glm::vec3* dev_normals = NULL;
static glm::vec3* dev_tangents = NULL;
static float* dev_fsigns = NULL;
static Primitive* dev_lights = NULL;
static PathSegment* dev_paths1 = NULL;
static PathSegment* dev_paths2 = NULL;
static ShadeableIntersection* dev_intersections1 = NULL;
static ShadeableIntersection* dev_intersections2 = NULL;

static PixelSensor* dev_pixelSensor = NULL;
static RGBFilm* dev_film = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene, Allocator alloc) {
	hst_scene = scene;

	dev_pixelSensor = alloc.new_object<PixelSensor>(RGBColorSpace::sRGB, nullptr, 0.03, alloc);

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	dev_film = alloc.new_object<RGBFilm>(dev_image, RGBColorSpace::sRGB, 100.0f);

	cudaMalloc(&dev_paths1, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_paths2, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_objs, scene->objects.size() * sizeof(Object));
	cudaMemcpy(dev_objs, scene->objects.data(), scene->objects.size() * sizeof(Object), cudaMemcpyHostToDevice);

	if (scene->triangles.size())
	{
		cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(glm::ivec3));
		cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(glm::ivec3), cudaMemcpyHostToDevice);

		cudaMalloc(&dev_vertices, scene->verticies.size() * sizeof(glm::vec3));
		cudaMemcpy(dev_vertices, scene->verticies.data(), scene->verticies.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

		cudaMalloc(&dev_uvs, scene->uvs.size() * sizeof(glm::vec2));
		cudaMemcpy(dev_uvs, scene->uvs.data(), scene->uvs.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);
		if (scene->normals.size())
		{
			cudaMalloc(&dev_normals, scene->normals.size() * sizeof(glm::vec3));
			cudaMemcpy(dev_normals, scene->normals.data(), scene->normals.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
		}
		if (scene->tangents.size())
		{
			cudaMalloc(&dev_tangents, scene->tangents.size() * sizeof(glm::vec3));
			cudaMemcpy(dev_tangents, scene->tangents.data(), scene->tangents.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
		}
		if (scene->fSigns.size())
		{
			cudaMalloc(&dev_fsigns, scene->fSigns.size() * sizeof(float));
			cudaMemcpy(dev_fsigns, scene->fSigns.data(), scene->fSigns.size() * sizeof(float), cudaMemcpyHostToDevice);
		}
	}

#if MTBVH
	cudaMalloc(&dev_mtbvhArray, scene->MTBVHArray.size() * sizeof(MTBVHGPUNode));
	cudaMemcpy(dev_mtbvhArray, scene->MTBVHArray.data(), scene->MTBVHArray.size() * sizeof(MTBVHGPUNode), cudaMemcpyHostToDevice);
#else
	cudaMalloc(&dev_bvhArray, scene->bvhArray.size() * sizeof(BVHGPUNode));
	cudaMemcpy(dev_bvhArray, scene->bvhArray.data(), scene->bvhArray.size() * sizeof(BVHGPUNode), cudaMemcpyHostToDevice);
#endif

	cudaMalloc(&dev_primitives, scene->primitives.size() * sizeof(Primitive));
	cudaMemcpy(dev_primitives, scene->primitives.data(), scene->primitives.size() * sizeof(Primitive), cudaMemcpyHostToDevice);

	if (scene->lights.size())
	{
		cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Primitive));
		cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Primitive), cudaMemcpyHostToDevice);
	}

	if (scene->materials.size())
	{
		cudaMalloc(&dev_materials, scene->materials.size() * sizeof(MaterialPtr));
		cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(MaterialPtr), cudaMemcpyHostToDevice);
	}
	
	if (scene->media.size())
	{
		cudaMalloc(&dev_media, scene->media.size() * sizeof(MediumPtr));
		cudaMemcpy(dev_media, scene->media.data(), scene->media.size() * sizeof(MediumPtr), cudaMemcpyHostToDevice);
	}


	cudaMalloc(&dev_intersections1, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections1, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_intersections2, pixelcount * sizeof(ShadeableIntersection));


#if !STOCHASTIC_SAMPLING && FIRST_INTERSECTION_CACHING
	cudaMalloc(&dev_intersectionCache, pixelcount * sizeof(ShadeableIntersection));
	cudaMalloc(&dev_pathCache, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_rayValidCache, pixelcount * sizeof(int));
	cudaMalloc(&dev_imageCache, pixelcount * sizeof(glm::vec3));
#endif
	// TODO: initialize any extra device memeory you need

	checkCUDAError("pathtraceInit");
}

void pathtraceFree(Scene* scene) {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths1);
	cudaFree(dev_paths2);
	cudaFree(dev_objs);
	if (scene->triangles.size())
	{
		cudaFree(dev_triangles);
		cudaFree(dev_vertices);
		cudaFree(dev_uvs);
		if (scene->normals.size())
		{
			cudaFree(dev_normals);
		}
		if (scene->tangents.size())
		{
			cudaFree(dev_tangents);
		}
		if (scene->fSigns.size())
		{
			cudaFree(dev_fsigns);
		}
	}
	cudaFree(dev_primitives);
	if (scene->lights.size())
	{
		cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Primitive));
		cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Primitive), cudaMemcpyHostToDevice);
	}
#if MTBVH
	cudaFree(dev_mtbvhArray);
#else
	cudaFree(dev_bvhArray);
#endif
	cudaFree(dev_materials);
	cudaFree(dev_media);
	cudaFree(dev_intersections1);
	cudaFree(dev_intersections2);
#if !STOCHASTIC_SAMPLING && FIRST_INTERSECTION_CACHING
	cudaFree(dev_intersectionCache);
	cudaFree(dev_pathCache);
	cudaFree(dev_rayValidCache);
	cudaFree(dev_imageCache);
#endif
	// TODO: clean up any extra device memory you created

	checkCUDAError("pathtraceFree");
}

__device__ inline glm::vec2 util_concentric_sample_disk(glm::vec2 rand)
{
	rand = 2.0f * rand - 1.0f;
	if (rand.x == 0 && rand.y == 0)
	{
		return glm::vec2(0);
	}
	const float pi_4 = PI / 4, pi_2 = PI / 2;
	bool x_g_y = abs(rand.x) > abs(rand.y);
	float theta = x_g_y ? pi_4 * rand.y / rand.x : pi_2 - pi_4 * rand.x / rand.y;
	float r = x_g_y ? rand.x : rand.y;
	return glm::vec2(cos(theta), sin(theta)) * r;
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, x * cam.resolution.y + y, 0);
	thrust::uniform_real_distribution<float> u01(0, 1);

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.transport = SampledSpectrum(1.0f);
		segment.lambda = SampledWavelengths::sample_visible(u01(rng));
		//segment.lambda = SampledWavelengths::sample_uniform(u01(rng));
#if STOCHASTIC_SAMPLING
		glm::vec2 jitter = glm::vec2(0.5f * (u01(rng) * 2.0f - 1.0f), 0.5f * (u01(rng) * 2.0f - 1.0f));
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + jitter[0])
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + jitter[1])
		);
#if DOF_ENABLED
		float lensR = cam.lensRadius;
		glm::vec3 perpDir = glm::cross(cam.right, cam.up);
		perpDir = glm::normalize(perpDir);
		float focalLen = cam.focalLength;
		float tFocus = focalLen / glm::abs(glm::dot(segment.ray.direction, perpDir));
		glm::vec2 offset = lensR * util_concentric_sample_disk(glm::vec2(u01(rng), u01(rng)));
		glm::vec3 newOri = offset.x * cam.right + offset.y * cam.up + cam.position;
		glm::vec3 pFocus = segment.ray.direction * tFocus + segment.ray.origin;
		segment.ray.direction = glm::normalize(pFocus - newOri);
		segment.ray.origin = newOri;
#endif

#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
		segment.lastMatPdf = -1;
		// TODO: change this to camera's medium
		segment.ray.medium = -1;
		segment.rng = rng;
	}
}



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

//__global__ void draw_gbuffer(
//	int num_paths
//	, PathSegment* pathSegments
//	, SceneInfoDev dev_sceneInfo
//	, SceneGbuffer gbuffer
//)
//{
//	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
//	if (path_index >= num_paths) return;
//	PathSegment& pathSegment = pathSegments[path_index];
//	Ray& ray = pathSegment.ray;
//	glm::vec3 rayDir = pathSegment.ray.direction;
//	glm::vec3 rayOri = pathSegment.ray.origin;
//	float x = fabs(rayDir.x), y = fabs(rayDir.y), z = fabs(rayDir.z);
//	int axis = x > y && x > z ? 0 : (y > z ? 1 : 2);
//	int sgn = rayDir[axis] > 0 ? 0 : 1;
//	int d = (axis << 1) + sgn;
//	const MTBVHGPUNode* currArray = dev_sceneInfo.dev_mtbvhArray + d * dev_sceneInfo.bvhDataSize;
//	int curr = 0;
//	ShadeableIntersection tmpIntersection;
//	tmpIntersection.t = 1e37f;
//	bool intersected = false;
//	while (curr >= 0 && curr < dev_sceneInfo.bvhDataSize)
//	{
//		bool outside = true;
//		float boxt = boundingBoxIntersectionTest(currArray[curr].bbox, ray, outside);
//		if (!outside) boxt = EPSILON;
//		if (boxt > 0 && boxt < tmpIntersection.t)
//		{
//			if (currArray[curr].startPrim != -1)//leaf node
//			{
//				int start = currArray[curr].startPrim, end = currArray[curr].endPrim;
//				bool intersect = util_bvh_leaf_intersect(start, end, dev_sceneInfo, ray, &tmpIntersection);
//				intersected = intersected || intersect;
//			}
//			curr = currArray[curr].hitLink;
//		}
//		else
//		{
//			curr = currArray[curr].missLink;
//		}
//	}
//	if (intersected)
//	{
//		int pixelIdx = pathSegment.pixelIndex;
//		gbuffer.dev_normal[pixelIdx] += tmpIntersection.surfaceNormal;
//		Material& mat = dev_sceneInfo.dev_materials[tmpIntersection.materialId];
//		glm::vec3 materialColor = mat.color;
//		if (mat.baseColorMap)
//		{
//			float4 color = tex2D<float4>(mat.baseColorMap, tmpIntersection.uv.x, tmpIntersection.uv.y);
//			materialColor.x = color.x;
//			materialColor.y = color.y;
//			materialColor.z = color.z;
//		}
//		gbuffer.dev_albedo[pixelIdx] += materialColor;
//	}
//	else
//	{
//		if (dev_sceneInfo.skyboxObj)
//		{
//			glm::vec2 uv = util_sample_spherical_map(glm::normalize(rayDir));
//			float4 skyColorRGBA = tex2D<float4>(dev_sceneInfo.skyboxObj, uv.x, uv.y);
//			glm::vec3 skyColor = glm::vec3(skyColorRGBA.x, skyColorRGBA.y, skyColorRGBA.z);
//			gbuffer.dev_albedo[pathSegment.pixelIndex] += skyColor;
//		}
//	}
//}


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
	// Set up the RNG
	// LOOK: this is how you use thrust's RNG! Please look at
	// makeSeededRandomEngine as well.
	thrust::default_random_engine& rng = pathSegments[idx].rng;
	thrust::uniform_real_distribution<float> u01(0, 1);
	// scattered in media
	if (intersection.materialId == -1)
	{
		rayValid[idx] = 1;
		return;
	}
	MaterialPtr material = materials[intersection.materialId];

//#if VIS_NORMAL
//	image[pathSegments[idx].pixelIndex] += (glm::normalize(intersection.surfaceNormal));
//	rayValid[idx] = 0;
//	return;
//#endif

	// If the material indicates that the object was a light, "light" the ray
	if (material.Is<EmissiveMaterial>()) {
		pathSegments[idx].transport *= material.Cast<EmissiveMaterial>()->Le(pathSegments[idx].lambda);
		rayValid[idx] = 0;
		if (!pathSegments[idx].transport.is_nan())
		{
			dev_film->add_radiance(sceneInfo.pixelSensor->to_sensor_rgb(pathSegments[idx].transport, pathSegments[idx].lambda), pathSegments[idx].pixelIndex);
		}
	}
	else {
		// For now if we encounter some non-emissive surface while rendering volumetrics, just error exit
		assert(sceneInfo.containsVolume == false);
		glm::vec3& woInWorld = pathSegments[idx].ray.direction;
		glm::vec3 nMap = glm::vec3(0, 0, 1);
		

		//if (material.normalMap != 0)
		//{
		//	float4 nMapCol = tex2D<float4>(material.normalMap, intersection.uv.x, intersection.uv.y);
		//	nMap.x = nMapCol.x;
		//	nMap.y = nMapCol.y;
		//	nMap.z = nMapCol.z;
		//	nMap = glm::pow(nMap, glm::vec3(1 / 2.2f));
		//	nMap = nMap * 2.0f - 1.0f;
		//	nMap = glm::normalize(nMap);
		//}
		glm::vec3 N = glm::normalize(intersection.surfaceNormal);
		glm::vec3 B, T;
		//if (material.normalMap != 0)
		//{
		//	T = intersection.surfaceTangent;
		//	T = glm::normalize(T - N * glm::dot(N, T));
		//	B = glm::cross(N, T);
		//	N = glm::normalize(T * nMap.x + B * nMap.y + N * nMap.z);
		//}
		//else
		//{
		math::Frame frame = math::Frame::from_z(N);
		//}
		glm::vec3 wo = frame.to_local(-woInWorld);
		wo = glm::normalize(wo);
		float pdf = 0;
		glm::vec3 wi;

		MaterialEvalInfo info(wo, intersection.uv, pathSegments[idx].lambda);

		BxDFPtr bxdf = material.get_bxdf(info, bxdfBufferLocal + threadIdx.x * BxDFMaxSize);

		SampledSpectrum f = bxdf.sample_f(wo, wi, pdf, rng);

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
		if (pdf > 0)
		{
			pathSegments[idx].transport *= f / pdf;
			glm::vec3 newDir = glm::normalize(frame.from_local(wi));
			glm::vec3 offset = glm::dot(newDir, N) < 0 ? -N : N;
			float offsetMult = !material.Is<DielectricMaterial>() ? SCATTER_ORIGIN_OFFSETMULT : SCATTER_ORIGIN_OFFSETMULT * 100.0f;
			pathSegments[idx].ray.origin = intersection.worldPos + offset * offsetMult;
			pathSegments[idx].ray.direction = newDir;
			rayValid[idx] = 1;
		}
		else
		{
			rayValid[idx] = 0;
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


//__global__ void addBackground(glm::vec3* dev_image, glm::vec3* dev_imageCache, int numPixels)
//{
//	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
//	if (index >= numPixels) return;
//	dev_image[index] += dev_imageCache[index];
//}



struct mat_comp {
	__host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
		return a.type < b.type;
	}
};

int compact_rays(int* rayValid,int* rayIndex,int numRays, bool sortByMat=false)
{
	thrust::device_ptr<PathSegment> dev_thrust_paths1(dev_paths1), dev_thrust_paths2(dev_paths2);
	thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections1(dev_intersections1), dev_thrust_intersections2(dev_intersections2);
	thrust::device_ptr<int> dev_thrust_rayValid(rayValid), dev_thrust_rayIndex(rayIndex);
	thrust::exclusive_scan(dev_thrust_rayValid, dev_thrust_rayValid + numRays, dev_thrust_rayIndex);
	int nextNumRays, tmp;
	cudaMemcpy(&tmp, rayIndex + numRays - 1, sizeof(int), cudaMemcpyDeviceToHost);
	nextNumRays = tmp;
	cudaMemcpy(&tmp, rayValid + numRays - 1, sizeof(int), cudaMemcpyDeviceToHost);
	nextNumRays += tmp;
	thrust::scatter_if(dev_thrust_paths1, dev_thrust_paths1 + numRays, dev_thrust_rayIndex, dev_thrust_rayValid, dev_thrust_paths2);
	thrust::scatter_if(dev_thrust_intersections1, dev_thrust_intersections1 + numRays, dev_thrust_rayIndex, dev_thrust_rayValid, dev_thrust_intersections2);
	if (sortByMat)
	{
		mat_comp cmp;
		thrust::sort_by_key(dev_thrust_intersections2, dev_thrust_intersections2 + nextNumRays, dev_thrust_paths2, cmp);
	}
	std::swap(dev_paths1, dev_paths2);
	std::swap(dev_intersections1, dev_intersections2);
	return nextNumRays;
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing
	SceneInfoDev dev_sceneInfo{};
	dev_sceneInfo.dev_materials = dev_materials;
	if (dev_media)
	{
		dev_sceneInfo.dev_media = dev_media;
		dev_sceneInfo.containsVolume = true;
	}
	dev_sceneInfo.dev_objs = dev_objs;
	dev_sceneInfo.objectsSize = hst_scene->objects.size();
	dev_sceneInfo.modelInfo.dev_triangles = dev_triangles;
	dev_sceneInfo.modelInfo.dev_vertices = dev_vertices;
	dev_sceneInfo.modelInfo.dev_normals = dev_normals;
	dev_sceneInfo.modelInfo.dev_uvs = dev_uvs;
	dev_sceneInfo.modelInfo.dev_tangents = dev_tangents;
	dev_sceneInfo.modelInfo.dev_fsigns = dev_fsigns;
	dev_sceneInfo.dev_primitives = dev_primitives;
#if USE_BVH
#if MTBVH
	dev_sceneInfo.dev_mtbvhArray = dev_mtbvhArray;
	dev_sceneInfo.bvhDataSize = hst_scene->MTBVHArray.size() / 6;
#else
	dev_sceneInfo.dev_bvhArray = dev_bvhArray;
	dev_sceneInfo.bvhDataSize = hst_scene->bvhTreeSize;
#endif
#endif // 
	dev_sceneInfo.skyboxObj = hst_scene->skyboxTextureObj;
	dev_sceneInfo.dev_lights = dev_lights;
	dev_sceneInfo.lightsSize = hst_scene->lights.size();

	dev_sceneInfo.pixelSensor = dev_pixelSensor;


	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, 32, dev_paths1);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths1 + pixelcount;
	int num_paths = dev_path_end - dev_paths1;
	int* rayValid, * rayIndex;
	
	int numRays = num_paths;
	cudaMalloc((void**)&rayValid, sizeof(int) * pixelcount);
	cudaMalloc((void**)&rayIndex, sizeof(int) * pixelcount);
	
	cudaDeviceSynchronize();
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	while (numRays && depth < 32) {

		// clean shading chunks
		depth++;
		cudaMemset(dev_intersections1, 0, pixelcount * sizeof(ShadeableIntersection));
		cudaMemset(rayValid, 0, sizeof(int) * pixelcount);
		dim3 numblocksPathSegmentTracing = (numRays + blockSize1d - 1) / blockSize1d;
#if !STOCHASTIC_SAMPLING && FIRST_INTERSECTION_CACHING
		if (iter != 1 && depth == 0)
		{
			cudaMemcpy(dev_intersections1, dev_intersectionCache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyHostToHost);
			cudaMemcpy(dev_paths1, dev_pathCache, pixelcount * sizeof(PathSegment), cudaMemcpyHostToHost);
			cudaMemcpy(rayValid, dev_rayValidCache, sizeof(int) * pixelcount, cudaMemcpyHostToHost);
			addBackground << < numblocksPathSegmentTracing, blockSize1d >> > (dev_image, dev_imageCache, pixelcount);
		}
		if (iter == 1||(iter!=1&&depth>0))
		{
#endif
			// tracing
			if (hst_scene->media.size() == 0)
			{
				compute_intersection_bvh_no_volume << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth
					, numRays
					, dev_paths1
					, dev_sceneInfo
					, dev_intersections1
					, rayValid
					, dev_film
					);
			}
			else
			{
				compute_intersection_bvh_volume_naive << <numblocksPathSegmentTracing, blockSize1d >> > (
					iter
					, depth
					, numRays
					, dev_paths1
					, dev_sceneInfo
					, dev_intersections1
					, rayValid
					, dev_film
					);
			}

#if !STOCHASTIC_SAMPLING && FIRST_INTERSECTION_CACHING
		}
		if (iter == 1 && depth == 0)
		{
			cudaMemcpy(dev_intersectionCache, dev_intersections1, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyHostToHost);
			cudaMemcpy(dev_pathCache, dev_paths1, pixelcount * sizeof(PathSegment), cudaMemcpyHostToHost);
			cudaMemcpy(dev_rayValidCache, rayValid, sizeof(int) * pixelcount, cudaMemcpyHostToHost);
			cudaMemcpy(dev_imageCache, dev_image, sizeof(glm::vec3) * pixelcount, cudaMemcpyHostToHost);
		}
#endif

		cudaDeviceSynchronize();
		checkCUDAError("compute_intersection");

		

#if SORT_BY_MATERIAL_TYPE
		numRays = compact_rays(rayValid, rayIndex, numRays, true);
#else
		numRays = compact_rays(rayValid, rayIndex, numRays);
#endif

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
		if (!numRays) break;
		dim3 numblocksLightScatter = (numRays + blockSize1d - 1) / blockSize1d;
#if USE_MIS
		scatter_on_intersection_mis << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			numRays,
			dev_intersections1,
			dev_paths1,
			dev_sceneInfo,
			rayValid,
			dev_image
			);
#else
		scatter_on_intersection << <numblocksPathSegmentTracing, blockSize1d , BxDFMaxSize * blockSize1d >> > (
			iter,
			numRays,
			dev_intersections1,
			dev_paths1,
			dev_sceneInfo,
			rayValid,
			dev_film
			);
#endif
		cudaDeviceSynchronize();
		checkCUDAError("scatter_on_intersection");

		numRays = compact_rays(rayValid, rayIndex, numRays);

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}


	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_film);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	cudaFree(rayValid);
	cudaFree(rayIndex);

	checkCUDAError("pathtrace");
}

//void DrawGbuffer(int numIter)
//{
//	if (!USE_BVH) throw;
//
//	const Camera& cam = hst_scene->state.camera;
//	const int pixelcount = cam.resolution.x * cam.resolution.y;
//
//	SceneInfoDev dev_sceneInfo{};
//	dev_sceneInfo.dev_materials = dev_materials;
//	dev_sceneInfo.dev_objs = dev_objs;
//	dev_sceneInfo.objectsSize = hst_scene->objects.size();
//	dev_sceneInfo.modelInfo.dev_triangles = dev_triangles;
//	dev_sceneInfo.modelInfo.dev_vertices = dev_vertices;
//	dev_sceneInfo.modelInfo.dev_normals = dev_normals;
//	dev_sceneInfo.modelInfo.dev_uvs = dev_uvs;
//	dev_sceneInfo.modelInfo.dev_tangents = dev_tangents;
//	dev_sceneInfo.modelInfo.dev_fsigns = dev_fsigns;
//	dev_sceneInfo.dev_primitives = dev_primitives;
//#if USE_BVH
//#if MTBVH
//	dev_sceneInfo.dev_mtbvhArray = dev_mtbvhArray;
//	dev_sceneInfo.bvhDataSize = hst_scene->MTBVHArray.size() / 6;
//#else
//	dev_sceneInfo.dev_bvhArray = dev_bvhArray;
//	dev_sceneInfo.bvhDataSize = hst_scene->bvhTreeSize;
//#endif
//#endif // 
//	dev_sceneInfo.skyboxObj = hst_scene->skyboxTextureObj;
//
//	const dim3 blockSize2d(8, 8);
//	const dim3 blocksPerGrid2d(
//		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
//		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
//	
//	const int blockSize1d = 128;
//	dim3 numblocksPathSegmentTracing = (pixelcount + blockSize1d - 1) / blockSize1d;
//	SceneGbuffer dev_gbuffer;
//	glm::vec3* dev_albedo,*dev_normal;
//	cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
//	cudaMalloc(&dev_normal, pixelcount * sizeof(glm::vec3));
//	dev_gbuffer.dev_albedo = dev_albedo;
//	dev_gbuffer.dev_normal = dev_normal;
//	for (int i = 0; i < numIter; i++)
//	{
//		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, i, MAX_DEPTH, dev_paths1);
//		draw_gbuffer << <numblocksPathSegmentTracing, blockSize1d >> > (pixelcount, dev_paths1, dev_sceneInfo, dev_gbuffer);
//	}
//	hst_scene->state.albedo.resize(pixelcount);
//	hst_scene->state.normal.resize(pixelcount);
//	cudaMemcpy(hst_scene->state.albedo.data(), dev_albedo, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
//	cudaMemcpy(hst_scene->state.normal.data(), dev_normal, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
//	cudaFree(dev_albedo);
//	cudaFree(dev_normal);
//	for (int i = 0; i < pixelcount; i++)
//	{
//		hst_scene->state.albedo[i] /= (float)numIter;
//		hst_scene->state.normal[i] /= (float)numIter;
//	}
//}
