#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"
#include "bvh.h"


// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__device__ inline glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}
/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__device__ inline  glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}
__device__ float util_geometry_ray_box_intersection(const glm::vec3& pMin, const glm::vec3& pMax, const Ray& r, bool* outside, glm::vec3* normal = nullptr);

__device__ bool util_geometry_ray_box_intersection(const glm::vec3& pMin, const glm::vec3& pMax, const Ray& r, float t_max, float* t_min_p, float* t_max_p);
// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__device__ float boxIntersectionTest(const Object& box, const Ray& r,
    glm::vec3& intersectionPoint, glm::vec3& normal);
__device__ inline float boundingBoxIntersectionTest(const BoundingBox& bbox, const Ray& r, bool& outside)
{
    return util_geometry_ray_box_intersection(bbox.pMin, bbox.pMax, r, &outside);
}
// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__device__ float util_geometry_ray_sphere_intersection(const Object& sphere, const Ray& r,
    glm::vec3& intersectionPoint, glm::vec3& normal);
__device__ bool util_geometry_ray_triangle_intersection(
    const glm::vec3& orig, const glm::vec3& dir,
    const glm::vec3& A, const glm::vec3& B, const glm::vec3& C,
    float& t, glm::vec3& normal, glm::vec3& baryCoord);
__device__ float triangleIntersectionTest(const ObjectTransform& Transform, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const Ray& r, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec3& baryCoord);
__device__ inline glm::vec2 util_sample_spherical_map(glm::vec3 v)
{
    const glm::vec2 invAtan = glm::vec2(0.1591, 0.3183);
    glm::vec2 uv = glm::vec2(atan2(v.z, v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5;
    return uv;
}
__device__ bool util_bvh_leaf_intersect(
    int primsStart,
    int primsEnd,
    const SceneInfoDev& dev_sceneInfo,
    Ray* ray,
    ShadeableIntersection* intersection
);
__device__ inline float util_bvh_leaf_test_intersect(
    int primsStart,
    int primsEnd,
    const SceneInfoDev& dev_sceneInfo,
    const Ray& ray
);
__device__ bool util_test_visibility(glm::vec3 p0, glm::vec3 p1, const SceneInfoDev& dev_sceneInfo);
__device__ bool util_bvh_test_visibility(glm::vec3 p0, glm::vec3 p1, const SceneInfoDev& dev_sceneInfo);