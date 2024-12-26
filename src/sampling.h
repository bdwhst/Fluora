#pragma once
#ifndef SAMPLING_H
#define SAMPLING_H
#include <cuda.h>
#include <glm/glm.hpp>

__device__ glm::vec2 util_math_uniform_sample_triangle(const glm::vec2& rand);

__device__ glm::vec3 util_sample_hemisphere_uniform(const glm::vec2& random);

__device__ glm::vec2 util_sample_disk_uniform(const glm::vec2& random);

__device__ glm::vec3 util_sample_hemisphere_cosine(const glm::vec2& random);


#endif // !SAMPLING_H
