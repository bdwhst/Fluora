#include "sampling.h"
#include "mathUtils.h"
__device__ glm::vec2 util_math_uniform_sample_triangle(const glm::vec2& rand)
{
	float t = sqrt(rand.x);
	return glm::vec2(1 - t, t * rand.y);
}

__device__ glm::vec3 util_sample_hemisphere_uniform(const glm::vec2& random)
{
	float z = random.x;
	float sq_1_z_2 = sqrt(math::max(1 - z * z, 0.0f));
	float phi = math::two_pi * random.y;
	return glm::vec3(cos(phi) * sq_1_z_2, sin(phi) * sq_1_z_2, z);
}

__device__ glm::vec2 util_sample_disk_uniform(const glm::vec2& random)
{
	float r = sqrt(random.x);
	float theta = math::two_pi * random.y;
	return glm::vec2(r * cos(theta), r * sin(theta));
}

__device__ glm::vec3 util_sample_hemisphere_cosine(const glm::vec2& random)
{
	glm::vec2 t = util_sample_disk_uniform(random);
	return glm::vec3(t.x, t.y, sqrt(1 - t.x * t.x - t.y * t.y));
}
