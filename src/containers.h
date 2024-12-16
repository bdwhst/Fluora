#pragma once
#include "memoryUtils.h"
#include "mathUtils.h"
#include <glm/glm.hpp>

class BundledParams
{
public:
	BundledParams() = default;

	void insert_float(const std::string& key, float value) { float_data.insert({ key,value }); }
	void insert_vec3(const std::string& key, const glm::vec3& value) { vec3_data.insert({ key,value }); }
	void insert_texture(const std::string& key, cudaTextureObject_t value) { tex_data.insert({ key,value }); }
	void insert_ptr(const std::string& key, void* value) { ptr_data.insert({ key,value }); }
	void insert_spectrum(const std::string& key, SpectrumPtr value) { spec_data.insert({ key,value }); }
	void insert_string(const std::string& key, const std::string& value) { str_data.insert({ key,value }); }
	float get_float(const std::string& key, float default_val = 0.0f) const
	{
		if (float_data.count(key))
		{
			return float_data.at(key);
		}
		return default_val;
	}
	glm::vec3 get_vec3(const std::string& key, const glm::vec3& default_val = glm::vec3(0.0f))  const
	{
		if (vec3_data.count(key))
		{
			return vec3_data.at(key);
		}
		return default_val;
	}
	cudaTextureObject_t get_texture(const std::string& key, cudaTextureObject_t default_val = 0) const
	{
		if (tex_data.count(key))
		{
			return tex_data.at(key);
		}
		return default_val;
	}
	void* get_ptr(const std::string& key, void* default_val = nullptr) const
	{
		if (ptr_data.count(key))
		{
			return ptr_data.at(key);
		}
		return default_val;
	}
	SpectrumPtr get_spec(const std::string& key, SpectrumPtr default_val = nullptr) const
	{
		if (spec_data.count(key))
		{
			return spec_data.at(key);
		}
		return default_val;
	}
	std::string get_string(const std::string& key, const std::string& default_val = "") const
	{
		if (str_data.count(key))
		{
			return str_data.at(key);
		}
		return default_val;
	}
	std::unordered_map<std::string, SpectrumPtr>& get_specs()
	{
		return spec_data;
	}
private:
	std::unordered_map<std::string, float> float_data;
	std::unordered_map<std::string, cudaTextureObject_t> tex_data;
	std::unordered_map<std::string, glm::vec3> vec3_data;
	std::unordered_map<std::string, void*> ptr_data;
	std::unordered_map<std::string, SpectrumPtr> spec_data;
	std::unordered_map<std::string, std::string> str_data;
};


template <typename T>
class SampledGrid
{
public:
	SampledGrid():nx(0),ny(0),nz(0),data(nullptr){}
	SampledGrid(float* v, int nx, int ny, int nz, Allocator alloc):nx(nx),ny(ny),nz(nz)
	{
		data = alloc.allocate<T>((uint64_t)nx * ny * nz);
		cudaMemcpy(data, v, sizeof(T) * nx * ny * nz);
	}
	__host__ __device__ operator bool() const
	{
		return data != nullptr;
	}
	template <typename F>
	__host__ __device__ auto look_up(const glm::vec3& p, F convert) const
	{
		glm::vec3 pSamples(p.x * nx - .5f, p.y * ny - .5f, p.z * nz - .5f);
		glm::ivec3 pi = pSamples;
		glm::vec3 d = pSamples - glm::vec3(pi);

		auto d00 = math::lerp(d.x, look_up(pi, convert), 
			look_up(pi + glm::ivec3(1, 0, 0), convert));
		auto d10 = math::lerp(d.x, look_up(pi + glm::ivec3(0, 1, 0), convert),
			look_up(pi + glm::ivec3(1, 1, 0), convert));
		auto d01 = math::lerp(d.x, look_up(pi + glm::ivec3(0, 0, 1), convert),
			look_up(pi + glm::ivec3(1, 0, 1), convert));
		auto d11 = math::lerp(d.x, look_up(pi + glm::ivec3(0, 1, 1), convert),
			look_up(pi + glm::ivec3(1, 1, 1), convert));
		return math::lerp(d.z, math::lerp(d.y, d00, d10), math::lerp(d.y, d01, d11));
	}
	__host__ __device__ T look_up(const glm::vec3& p) const
	{
		glm::vec3 pSamples(p.x * nx - .5f, p.y * ny - .5f, p.z * nz - .5f);
		glm::ivec3 pi = pSamples;
		glm::vec3 d = pSamples - glm::vec3(pi);

		auto d00 = math::lerp(d.x, look_up(pi),
			look_up(pi + glm::ivec3(1, 0, 0)));
		auto d10 = math::lerp(d.x, look_up(pi + glm::ivec3(0, 1, 0)),
			look_up(pi + glm::ivec3(1, 1, 0)));
		auto d01 = math::lerp(d.x, look_up(pi + glm::ivec3(0, 0, 1)),
			look_up(pi + glm::ivec3(1, 0, 1)));
		auto d11 = math::lerp(d.x, look_up(pi + glm::ivec3(0, 1, 1)),
			look_up(pi + glm::ivec3(1, 1, 1)));
		return math::lerp(d.z, math::lerp(d.y, d00, d10), math::lerp(d.y, d01, d11));
	}
	template <typename F>
	__host__ __device__ auto look_up(const glm::ivec3& p, F convert) const
	{
		if (!(p.x >= 0 && p.x < nx && p.y >= 0 && p.y < ny && p.z >= 0 && p.z < nz))
		{
			return convert(T{});
		}
		return convert(data[(p.z * ny + p.y) * nx + p.x]);
	}
	__host__ __device__ T look_up(const glm::ivec3& p) const
	{
		if (!(p.x >= 0 && p.x < nx && p.y >= 0 && p.y < ny && p.z >= 0 && p.z < nz))
		{
			return T{};
		}
		return data[(p.z * ny + p.y) * nx + p.x];
	}

	template <typename F>
	__host__ auto max_value(const glm::vec3& pMin, const glm::vec3& pMax, F convert) const -> decltype(convert(T{})) {
		glm::vec3 ps[2] = { glm::vec3(pMin.x * nx - .5f, pMin.y * ny - .5f,
								 pMin.z * nz - .5f),
						 glm::vec3(pMax.x * nx - .5f, pMax.y * ny - .5f,
								 pMax.z * nz - .5f) };
		glm::ivec3 pi[2] = { glm::max(glm::ivec3(glm::floor(ps[0])), glm::ivec3(0, 0, 0)),
						 glm::min(glm::ivec3(glm::floor(ps[1])) + glm::ivec3(1, 1, 1),
							 glm::ivec3(nx - 1, ny - 1, nz - 1)) };
		static_assert(!std::is_same<decltype(convert(T{})), void > ::value, "convert returns type void");
		auto maxValue = look_up(glm::ivec3(pi[0]), convert);
		for (int z = pi[0].z; z <= pi[1].z; ++z)
			for (int y = pi[0].y; y <= pi[1].y; ++y)
				for (int x = pi[0].x; x <= pi[1].x; ++x)
					maxValue = math::max(maxValue, look_up(glm::ivec3(x, y, z), convert));

		return maxValue;
	}
	__host__ T max_value(const glm::vec3& pMin, const glm::vec3& pMax) const {
		auto convert = [] __host__ __device__ (T value) -> T { return value; };
		
		return max_value(pMin, pMax, convert);
	}
private:
	int nx, ny, nz;
	T* data;
};