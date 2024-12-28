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
#include "lightSamplers.h"
#include "naiveIntegrator.h"
#include "misIntegrator.h"
//#include "materials.h"

IntegratorType mainIntegratorType = IntegratorType::mis;

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
		if (math::is_nan(pix))
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
static LightPtr* dev_lights = NULL;
static PathSegment* dev_paths1 = NULL;
static PathSegment* dev_paths2 = NULL;
static ShadeableIntersection* dev_intersections1 = NULL;
static ShadeableIntersection* dev_intersections2 = NULL;
static ShadowRaySegment* dev_shadowRayPaths = NULL;

static PixelSensor* dev_pixelSensor = NULL;
static RGBFilm* dev_film = NULL;
static LightSamplerPtr dev_lightSampler = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene, Allocator alloc) {
	//TODO: resolve mem leak by track every allocation, and avoid unnecessary resource recreation while moving camera

	hst_scene = scene;

	//DenselySampledSpectrum dIllum = spec::D(6500.f, {});
	dev_pixelSensor = alloc.new_object<PixelSensor>(RGBColorSpace::ACES2065_1, nullptr, 0.03, alloc);
	

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	dev_film = alloc.new_object<RGBFilm>(dev_pixelSensor, dev_image, RGBColorSpace::sRGB, 1e6f);

	cudaMalloc(&dev_paths1, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_paths2, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_shadowRayPaths, pixelcount * sizeof(ShadowRaySegment));

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
		cudaMalloc(&dev_lights, scene->lights.size() * sizeof(LightPtr));
		cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(LightPtr), cudaMemcpyHostToDevice);

		dev_lightSampler = alloc.new_object<UniformLightSampler>(dev_lights, scene->lights.size(), scene->skyboxLight);
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

	if (mainIntegratorType == IntegratorType::naive)
	{
		guiData->integratorType = "naive";
	}
	else
	{
		guiData->integratorType = "mis";
	}
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
		cudaFree(dev_lights);
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
		segment.r_u = SampledSpectrum(1.0f);
		segment.r_l = SampledSpectrum(1.0f);
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

__global__ void init_atomics()
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		numShadowRays.store(0);
	}
}

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
	/*if (sortByMat)
	{
		mat_comp cmp;
		thrust::sort_by_key(dev_thrust_intersections2, dev_thrust_intersections2 + nextNumRays, dev_thrust_paths2, cmp);
	}*/
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
	/*dev_sceneInfo.dev_lights = dev_lights;
	dev_sceneInfo.lightsSize = hst_scene->lights.size();*/

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

	
	while (numRays && depth < MAX_DEPTH) {

		// clean shading chunks
		cudaMemset(dev_intersections1, 0, pixelcount * sizeof(ShadeableIntersection));
		cudaMemset(rayValid, 0, sizeof(int) * pixelcount);
		if (mainIntegratorType == IntegratorType::mis)
		{
			// clear num of shadow rays
			init_atomics << <1, 1 >> > ();
			cudaDeviceSynchronize();
		}

		dim3 numblocksPathSegmentTracing = (numRays + blockSize1d - 1) / blockSize1d;
#if !STOCHASTIC_SAMPLING && FIRST_INTERSECTION_CACHING
		if (iter != 1 && depth == 0)
		{
			cudaMemcpy(dev_intersections1, dev_intersectionCache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyHostToHost);
			cudaMemcpy(dev_paths1, dev_pathCache, pixelcount * sizeof(PathSegment), cudaMemcpyHostToHost);
			cudaMemcpy(rayValid, dev_rayValidCache, sizeof(int) * pixelcount, cudaMemcpyHostToHost);
			addBackground << < numblocksPathSegmentTracing, blockSize1d >> > (dev_image, dev_imageCache, pixelcount);
		}
		if (iter == 1 || (iter != 1 && depth > 0))
		{
#endif
			// tracing
			if (hst_scene->media.size() == 0)
			{
				if (mainIntegratorType == IntegratorType::naive)
				{
					compute_intersection_bvh_no_volume << <numblocksPathSegmentTracing, blockSize1d >> > (
						depth
						, numRays
						, dev_paths1
						, dev_sceneInfo
						, dev_intersections1
						, rayValid
						, dev_film
						, hst_scene->skyboxLight
						);
				}
				else if (mainIntegratorType == IntegratorType::mis)
				{
					compute_intersection_bvh_no_volume_mis << <numblocksPathSegmentTracing, blockSize1d >> > (
						depth
						, numRays
						, dev_paths1
						, dev_sceneInfo
						, dev_intersections1
						, rayValid
						, dev_film
						, dev_lightSampler
						);
				}
			}
			else
			{
				if (mainIntegratorType == IntegratorType::naive)
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
						, hst_scene->skyboxLight
						);
				}
				else if (mainIntegratorType == IntegratorType::mis)
				{
					compute_intersection_bvh_volume_mis << <numblocksPathSegmentTracing, blockSize1d >> > (
						iter
						, depth
						, numRays
						, dev_paths1
						, dev_sceneInfo
						, dev_intersections1
						, rayValid
						, dev_film
						, dev_lightSampler
						, dev_shadowRayPaths
						);
				}
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
		if (!numRays) break;
		dim3 numblocksLightScatter = (numRays + blockSize1d - 1) / blockSize1d;
		if (mainIntegratorType == IntegratorType::naive)
		{
			scatter_on_intersection << <numblocksPathSegmentTracing, blockSize1d, BxDFMaxSize* blockSize1d >> > (
				iter,
				numRays,
				dev_intersections1,
				dev_paths1,
				dev_sceneInfo,
				rayValid,
				dev_film
				);
		}
		else if(mainIntegratorType == IntegratorType::mis)
		{
			scatter_on_intersection_mis << <numblocksPathSegmentTracing, blockSize1d, BxDFMaxSize* blockSize1d >> > (
				iter,
				depth,
				numRays,
				dev_intersections1,
				dev_paths1,
				dev_sceneInfo,
				rayValid,
				dev_film,
				dev_lightSampler,
				dev_shadowRayPaths
				);
		}
		cudaDeviceSynchronize();
		checkCUDAError("scatter_on_intersection");

		numRays = compact_rays(rayValid, rayIndex, numRays);
		if (mainIntegratorType == IntegratorType::mis)
		{
			// mis light sampling
			int currNumShadowRays = 0;
			cudaMemcpyFromSymbol(&currNumShadowRays, numShadowRays, sizeof(int), 0, cudaMemcpyDeviceToHost);
			if (currNumShadowRays > 0)
			{
				dim3 numblocksShadowRay = (currNumShadowRays + blockSize1d - 1) / blockSize1d;
				if (hst_scene->media.size() == 0)
				{
					sample_Ld << < numblocksShadowRay, blockSize1d >> > (
						currNumShadowRays,
						dev_shadowRayPaths,
						dev_lightSampler,
						dev_sceneInfo,
						dev_film
						);
				}
				else
				{
					sample_Ld_volume << < numblocksShadowRay, blockSize1d >> > (
						currNumShadowRays,
						dev_shadowRayPaths,
						dev_lightSampler,
						dev_sceneInfo,
						dev_film
						);
				}
			}
		}

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth + 1;
		}
		depth++;
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

