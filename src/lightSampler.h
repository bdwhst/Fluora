#pragma once
#include "light.h"
#include "taggedptr.h"
struct SampledLight {
    LightPtr light;
    float pdf = 0;
    __device__ __host__ SampledLight(): light(nullptr), pdf(0){}
};

class UniformLightSampler;
class LightSamplerPtr :public TaggedPointer<UniformLightSampler>
{
public:
	using TaggedPointer::TaggedPointer;
    __device__ SampledLight sample(const LightSampleContext& ctx, float u) const;
    __device__ SampledLight sample(float u) const;
    __device__ float pmf(const LightSampleContext& ctx, LightPtr light);
    __device__ float pmf(LightPtr light);
    __device__ LightPtr get_light(int idx) const;
    __device__ LightPtr get_infinite_light() const;
    __device__ bool have_infinite_light() const;
};


