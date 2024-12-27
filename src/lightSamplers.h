#pragma once

#include "lightSampler.h"

class UniformLightSampler
{
public:
	UniformLightSampler(LightPtr* lights, int lightsSize, LightPtr infiniteLight = nullptr):m_lights(lights), m_lightsSize(lightsSize), m_infiniteLight(infiniteLight){}
	GPU_FUNC SampledLight sample(const LightSampleContext& ctx, float u) const
	{
		return sample(u);
	}
	GPU_FUNC SampledLight sample(float u) const
	{
		SampledLight sampledLight;
		sampledLight.pdf = 1.0f / m_lightsSize;
		sampledLight.light = m_lights[math::clamp((int)(u * m_lightsSize), 0, m_lightsSize - 1)];
		return sampledLight;
	}
	GPU_FUNC float pmf(const LightSampleContext& ctx, LightPtr light)
	{
		return pmf(light);
	}
	GPU_FUNC float pmf(LightPtr light)
	{
		return 1.0f / m_lightsSize;
	}
	GPU_FUNC LightPtr get_light(int idx) const
	{
		assert(idx >= 0 && idx < m_lightsSize);
		return m_lights[idx];
	}

	GPU_FUNC LightPtr get_infinite_light() const
	{
		return m_infiniteLight;
	}

	GPU_FUNC bool have_infinite_light() const
	{
		return m_infiniteLight != nullptr;
	}
private:
	LightPtr* m_lights;
	LightPtr m_infiniteLight;
	int m_lightsSize;
};