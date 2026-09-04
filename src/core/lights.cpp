#include "lights.h"

#include <algorithm>

std::vector<RtLight> buildLightList(const CoreScene& scene, bool hasEnvMap)
{
    std::vector<RtLight> lights;
    if (hasEnvMap)
        lights.push_back(RtLight{ RT_LIGHT_ENV, 0, 0, 0 });
    auto emissive = [&](int matId) {
        return matId >= 0 && matId < (int)scene.materials.size()
            && scene.materials[matId].type == CoreMaterialType::Emissive;
    };
    for (size_t i = 0; i < scene.objects.size(); i++) {
        const CoreObject& o = scene.objects[i];
        if (emissive(o.materialId))
            lights.push_back(RtLight{ o.geomType == CORE_GEOM_SPHERE ? RT_LIGHT_SPHERE : RT_LIGHT_CUBE,
                                      (uint32_t)i, 0, 0 });
    }
    for (size_t t = 0; t < scene.tris.size(); t++) {
        if (emissive((int)scene.tris[t].w))
            lights.push_back(RtLight{ RT_LIGHT_TRI, (uint32_t)t, 0, 0 });
    }
    return lights;
}

namespace {

// Distribution1D constructor (sampling.h): cdf[0]=0, cdf[i] = cdf[i-1] + f[i-1]/n,
// funcInt = cdf[n]; normalized, or linear when funcInt == 0.
float buildDist1D(const float* f, uint32_t n, float* cdf)
{
    cdf[0] = 0.0f;
    for (uint32_t i = 1; i <= n; i++)
        cdf[i] = cdf[i - 1] + f[i - 1] / (float)n;
    float funcInt = cdf[n];
    if (funcInt == 0.0f) {
        for (uint32_t i = 1; i <= n; i++)
            cdf[i] = (float)i / (float)n;
    } else {
        for (uint32_t i = 1; i <= n; i++)
            cdf[i] /= funcInt;
    }
    return funcInt;
}

} // namespace

std::vector<float> buildEnvDistribution(const HdrImage& img, const glm::vec3& maxRadiance)
{
    const uint32_t nu = (uint32_t)img.width, nv = (uint32_t)img.height;
    std::vector<float> d(ENVDIST_FLOATS(nu, nv), 0.0f);
    float* condFunc = d.data() + ENVDIST_OFF_COND_FUNC(nu, nv);
    float* condCdf = d.data() + ENVDIST_OFF_COND_CDF(nu, nv);
    float* condInt = d.data() + ENVDIST_OFF_COND_INT(nu, nv);
    float* margFunc = d.data() + ENVDIST_OFF_MARG_FUNC(nu, nv);
    float* margCdf = d.data() + ENVDIST_OFF_MARG_CDF(nu, nv);
    for (uint32_t v = 0; v < nv; v++) {
        for (uint32_t u = 0; u < nu; u++) {
            const float* px = &img.rgba[((size_t)v * nu + u) * 4];
            glm::vec3 rgb = glm::min(glm::vec3(px[0], px[1], px[2]), maxRadiance);
            condFunc[v * nu + u] = 0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
        }
        condInt[v] = buildDist1D(condFunc + v * nu, nu, condCdf + v * (nu + 1));
        margFunc[v] = condInt[v];
    }
    d[ENVDIST_OFF_MARG_INT(nu, nv)] = buildDist1D(margFunc, nv, margCdf);
    return d;
}

void applyEnvScale(HdrImage& img, float scale, const glm::vec3& maxRadiance)
{
    for (size_t i = 0; i + 3 < img.rgba.size(); i += 4) {
        img.rgba[i + 0] = std::min(img.rgba[i + 0] * scale, maxRadiance.r);
        img.rgba[i + 1] = std::min(img.rgba[i + 1] * scale, maxRadiance.g);
        img.rgba[i + 2] = std::min(img.rgba[i + 2] * scale, maxRadiance.b);
    }
}
