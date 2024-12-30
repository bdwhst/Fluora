#pragma once
#include <cuda.h>
#include "sceneStructs.h"
#include "light.h"

__global__ void compute_intersection_bvh_no_volume(
	int num_paths
	, PathSegment* pathSegments
	, SceneInfoDev dev_sceneInfo
	, ShadeableIntersection* intersections
	, int* rayValid
	, RGBFilm* dev_film
	, LightPtr dev_skyboxLight
);

// Does not handle surface intersection
__global__ void compute_intersection_bvh_volume_naive(
	int iter
	, int num_paths
	, PathSegment* pathSegments
	, SceneInfoDev dev_sceneInfo
	, ShadeableIntersection* intersections
	, int* rayValid
	, RGBFilm* dev_film
	, LightPtr dev_skyboxLight
);


__global__ void scatter_on_intersection(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, SceneInfoDev sceneInfo
	, int* rayValid
	, RGBFilm* dev_film
);
