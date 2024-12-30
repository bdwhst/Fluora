#pragma once
#include <cuda.h>
#include <cuda/atomic>
#include "sceneStructs.h"
#include "lightSampler.h"

__global__ void compute_intersection_bvh_no_volume_mis(
	int num_paths
	, PathSegment* pathSegments
	, SceneInfoDev dev_sceneInfo
	, ShadeableIntersection* intersections
	, int* rayValid
	, RGBFilm* dev_film
	, LightSamplerPtr lightSampler
);

__global__ void sample_Ld_volume(
	int numRays,
	ShadowRaySegment* shadowRaySegments,
	LightSamplerPtr lightSampler,
	SceneInfoDev sceneInfo,
	RGBFilm* dev_film
);

__global__ void compute_intersection_bvh_volume_mis(
	int iter
	, int num_paths
	, PathSegment* pathSegments
	, SceneInfoDev dev_sceneInfo
	, ShadeableIntersection* intersections
	, int* rayValid
	, RGBFilm* dev_film
	, LightSamplerPtr lightSampler
	, ShadowRaySegment* shadowRaySegments
);

extern __device__ cuda::atomic<int, cuda::thread_scope_device> numShadowRays;

__global__ void sample_Ld(
	int numRays, 
	ShadowRaySegment* shadowRaySegments, 
	LightSamplerPtr lightSampler, 
	SceneInfoDev sceneInfo, 
	RGBFilm* dev_film
);

__global__ void scatter_on_intersection_mis(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, SceneInfoDev sceneInfo
	, int* rayValid
	, RGBFilm* dev_film
	, LightSamplerPtr lightSampler
	, ShadowRaySegment* shadowRaySegments
);

__global__ void scatter_on_intersection_volume_mis(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, SceneInfoDev sceneInfo
	, int* rayValid
	, RGBFilm* dev_film
	, LightSamplerPtr lightSampler
	, ShadowRaySegment* shadowRaySegments
);

