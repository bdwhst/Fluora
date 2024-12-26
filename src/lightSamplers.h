#pragma once

#include "lightSampler.h"

class UniformLightSampler
{
public:
	UniformLightSampler(LightPtr* lights, int lightsSize):m_lights(lights), m_lightsSize(lightsSize){}
	__device__ SampledLight sample(const LightSampleContext& ctx, float u) const
	{
		return sample(u);
	}
	__device__ SampledLight sample(float u) const
	{
		SampledLight sampledLight;
		sampledLight.pdf = 1.0f / m_lightsSize;
		sampledLight.light = m_lights[math::clamp((int)(u * m_lightsSize), 0, m_lightsSize - 1)];
		return sampledLight;
	}
	__device__ float pmf(const LightSampleContext& ctx, LightPtr light)
	{
		return pmf(light);
	}
	__device__ float pmf(LightPtr light)
	{
		return 1.0f / m_lightsSize;
	}
	__device__ LightPtr get_light(int idx) const
	{
		assert(idx >= 0 && idx < m_lightsSize);
		return m_lights[idx];
	}
private:
	LightPtr* m_lights;
	int m_lightsSize;
};