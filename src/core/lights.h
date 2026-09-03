#pragma once
// Host-side light list + env-map sampling distribution for the portable
// renderer (device counterpart: light_shared.h). Backend-neutral: produces
// flat arrays the app uploads as buffers.
#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

#include "light_shared.h"
#include "scene_loader.h"
#include "image_loader.h"

// One RtLight per emissive analytic object and per emissive triangle (in the
// scene's post-BVH triangle order), plus an RT_LIGHT_ENV entry first when
// the scene has an environment map. Empty when the scene has no lights.
std::vector<RtLight> buildLightList(const CoreScene& scene, bool hasEnvMap);

// Distribution2D over the env texels' luminance (rgb clamped to maxRadiance),
// laid out per the ENVDIST_* offsets in light_shared.h. nu = width, nv = height.
std::vector<float> buildEnvDistribution(const HdrImage& img, const glm::vec3& maxRadiance);
