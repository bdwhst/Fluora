#include "lightSamplers.h"

__device__ SampledLight LightSamplerPtr::sample(const LightSampleContext& ctx, float u) const
{
	auto f = [&](auto ptr)
		{
			return ptr->sample(ctx, u);
		};
	return Dispatch(f);
}
__device__ SampledLight LightSamplerPtr::sample(float u) const
{
	auto f = [&](auto ptr)
		{
			return ptr->sample(u);
		};
	return Dispatch(f);
}
__device__ float LightSamplerPtr::pmf(const LightSampleContext& ctx, LightPtr light)
{
	auto f = [&](auto ptr)
		{
			return ptr->pmf(ctx, light);
		};
	return Dispatch(f);
}
__device__ float LightSamplerPtr::pmf(LightPtr light)
{
	auto f = [&](auto ptr)
		{
			return ptr->pmf(light);
		};
	return Dispatch(f);
}

__device__ __device__ LightPtr LightSamplerPtr::get_light(int idx) const
{
	auto f = [&](auto ptr)
		{
			return ptr->get_light(idx);
		};
	return Dispatch(f);
}

__device__ __device__ LightPtr LightSamplerPtr::get_infinite_light() const
{
	auto f = [&](auto ptr)
		{
			return ptr->get_infinite_light();
		};
	return Dispatch(f);
}

__device__ __device__ bool LightSamplerPtr::have_infinite_light() const
{
	auto f = [&](auto ptr)
		{
			return ptr->have_infinite_light();
		};
	return Dispatch(f);
}