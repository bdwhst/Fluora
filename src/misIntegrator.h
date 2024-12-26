#pragma once
#include <cuda.h>
#include <cuda/atomic>
#include "sceneStructs.h"
#include "lightSampler.h"

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
	, int depth
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, SceneInfoDev sceneInfo
	, int* rayValid
	, RGBFilm* dev_film
	, LightSamplerPtr lightSampler
	, ShadowRaySegment* shadowRaySegments
);

