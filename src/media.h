#pragma once
#include "taggedptr.h"
#include "spectrum.h"
#include "containers.h"
#include "medium.h"
#include "sceneStructs.h"

#include <nanovdb/NanoVDB.h>
#define NANOVDB_USE_ZIP 1
#include <nanovdb/util/IO.h>

#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/SampleFromVoxels.h>

#include <nanovdb/util/CudaDeviceBuffer.h>

class HomogeneousMajorantIterator
{
public:
	__device__ HomogeneousMajorantIterator() :called(true) {}
	__device__ HomogeneousMajorantIterator(float t_min, float t_max, SampledSpectrum sigma_maj) :called(false), t_min(t_min), t_max(t_max), sigma_maj(sigma_maj) {}
	__device__ bool next(float* t_min_p, float* t_max_p, SampledSpectrum* sigma_maj_p)
	{
		if (called) return false;
		called = true;
		*t_min_p = t_min;
		*t_max_p = t_max;
		*sigma_maj_p = sigma_maj;
		return true;
	}
	bool called;
	float t_min, t_max;
	SampledSpectrum sigma_maj;
};

struct MajorantGrid
{
public:
	MajorantGrid() = default;
	MajorantGrid(const glm::vec3& bmin, const glm::vec3& bmax, const glm::ivec3& res, Allocator alloc) :bmin(bmin), bmax(bmax), voxels(nullptr), res(res)
	{
		voxels = alloc.allocate<float>((uint64_t)res.x * res.y * res.z);
	}
	__device__ __host__ float look_up(int x, int y, int z) const
	{
		assert(x >= 0 && x < res.x && y >= 0 && y < res.y && z >= 0 && z < res.z);
		return voxels[x + res.x * y + res.x * res.y * z];
	}
	__host__ void set(int x, int y, int z, float v)
	{
		assert(x >= 0 && x < res.x && y >= 0 && y < res.y && z >= 0 && z < res.z);
		voxels[x + res.x * y + res.x * res.y * z] = v;
	}
	__host__ void get_voxel_bounds(int x, int y, int z, glm::vec3* bmin, glm::vec3* bmax) const
	{
		*bmin = glm::vec3(float(x) / res.x, float(y) / res.y, float(z) / res.z);
		*bmax = glm::vec3(float(x + 1) / res.x, float(y + 1) / res.y, float(z + 1) / res.z);
	}
	glm::vec3 bmin, bmax;
	float* voxels;
	glm::ivec3 res;
};

class DDAMajorantIterator
{
public:
	DDAMajorantIterator() = default;
	__device__ DDAMajorantIterator(const Ray& ray, float t_min, float t_max, const MajorantGrid* grid, SampledSpectrum sigma_t) :
		t_min(t_min), t_max(t_max), grid(grid), sigma_t(sigma_t)
	{
		const glm::vec3& ray_ori = ray.origin;
		const glm::vec3& ray_dir = ray.direction;
		using namespace math;
		glm::vec3 diag = grid->bmax - grid->bmin;
		glm::vec3 ray_ori_0 = (ray_ori - grid->bmin) / diag;
		glm::vec3 ray_dir_0 = ray_dir / diag;
		glm::vec3 gridIntersect = ray_ori_0 + ray_dir_0 * t_min;
		for (int axis = 0; axis < 3; ++axis)
		{
			voxel[axis] = math::clamp(gridIntersect[axis] * grid->res[axis], 0, grid->res[axis] - 1);
			delta_t[axis] = 1 / (abs(ray_dir_0[axis]) * grid->res[axis]);
			if (ray_dir_0[axis] == -0.0f)
			{
				ray_dir_0[axis] = 0.0f;
			}
			if (ray_dir_0[axis] >= 0)
			{
				float nextVoxelPos = float(voxel[axis] + 1) / grid->res[axis];
				next_crossing_t[axis] = t_min + (nextVoxelPos - gridIntersect[axis]) / ray_dir_0[axis];
				step[axis] = 1;
				voxel_limit[axis] = grid->res[axis];
			}
			else
			{
				float nextVoxelPos = float(voxel[axis]) / grid->res[axis];
				next_crossing_t[axis] = t_min + (nextVoxelPos - gridIntersect[axis]) / ray_dir_0[axis];
				step[axis] = -1;
				voxel_limit[axis] = -1;
			}
		}
	}
	__device__ bool next(float* t_min_p, float* t_max_p, SampledSpectrum* sigma_maj_p)
	{
		if (t_min >= t_max) return false;
		int bits = ((next_crossing_t[0] < next_crossing_t[1]) << 2) + ((next_crossing_t[0] < next_crossing_t[2]) << 1) + (next_crossing_t[1] < next_crossing_t[2]);
		const int cmpToAxis[8] = { 2,1,2,1,2,2,0,0 };
		int stepAxis = cmpToAxis[bits];
		float tVoxelExit = math::min(t_max, next_crossing_t[stepAxis]);
		SampledSpectrum sigma_maj = sigma_t * grid->look_up(voxel[0], voxel[1], voxel[2]);
		*t_min_p = t_min;
		*t_max_p = tVoxelExit;
		*sigma_maj_p = sigma_maj;

		t_min = tVoxelExit;
		if (next_crossing_t[stepAxis] > t_max) t_min = t_max;
		voxel[stepAxis] += step[stepAxis];
		if (voxel[stepAxis] == voxel_limit[stepAxis]) t_min = t_max;
		next_crossing_t[stepAxis] += delta_t[stepAxis];
		
		return true;
	}
private:
	SampledSpectrum sigma_t;
	float t_min = 1e38f, t_max = -1e38f;
	const MajorantGrid* grid;
	float next_crossing_t[3], delta_t[3];
	int step[3], voxel_limit[3], voxel[3];

};

class HomogeneousMedium
{
public:
	static HomogeneousMedium* create(const BundledParams& params, Allocator alloc);
	HomogeneousMedium(SpectrumPtr sigma_a, SpectrumPtr sigma_s, float sigma_scale, SpectrumPtr Le, float Le_scale, float g, Allocator alloc): sigma_a_spec(sigma_a, alloc), sigma_s_spec(sigma_s, alloc), Le_spec(Le, alloc), phase(g)
	{
		sigma_a_spec.scale(sigma_scale);
		sigma_s_spec.scale(sigma_scale);
		Le_spec.scale(Le_scale);
	}
	__device__ bool is_emissive() const
	{
		return Le_spec.max_value() > 0;
	}
	__device__ MediumProperties sample_point(const glm::vec3& point, const SampledWavelengths& lambda) const
	{
		SampledSpectrum sigma_a = sigma_a_spec.sample(lambda);
		SampledSpectrum sigma_s = sigma_s_spec.sample(lambda);
		SampledSpectrum Le = Le_spec.sample(lambda);
		return MediumProperties{ sigma_a, sigma_s, &phase, Le };
	}
	using MajorantIterator = HomogeneousMajorantIterator;
	__device__ MajorantIterator sample_ray(const Ray& ray, float t_max, const SampledWavelengths& lambda) const
	{
		SampledSpectrum sigma_a = sigma_a_spec.sample(lambda);
		SampledSpectrum sigma_s = sigma_s_spec.sample(lambda);

		return { 0, t_max, sigma_a + sigma_s };
	}
private:
	DenselySampledSpectrum sigma_a_spec, sigma_s_spec, Le_spec;
	HGPhaseFunction phase;
};








class GridMedium
{
public:
	GridMedium(const glm::vec3& bmin, const glm::vec3& bmax, const glm::mat4& render_from_medium, const glm::mat4& medium_from_render, SpectrumPtr sigma_a, SpectrumPtr sigma_s, float sigma_scale, float g, SampledGrid<float> d, SampledGrid<float> temperature, SpectrumPtr Le, SampledGrid<float> LeGrid, Allocator alloc);
	__device__ MediumProperties sample_point(const glm::vec3& point, const SampledWavelengths& lambda) const
	{
		SampledSpectrum sigma_a = sigma_a_spec.sample(lambda);
		SampledSpectrum sigma_s = sigma_s_spec.sample(lambda);
		glm::vec3 p = glm::vec3(medium_from_render * glm::vec4(point, 1.0f));
		p = (p - bmin) / (bmax - bmin);
		float d = density_grid.look_up(p);
		sigma_a *= d;
		sigma_s *= d;
		SampledSpectrum Le(0.0f);
		if (isEmissive)
		{
			float scale = Le_scale.look_up(p);
			if (scale > 0.0f)
			{
				if (temperature_grid)
				{
					//TODO: black body
				}
				else
				{
					Le = scale * Le_spec.sample(lambda);
				}
			}
		}
		return { sigma_a ,sigma_s, &phase, Le };
	}
	__device__ bool is_emissive() const
	{
		return isEmissive;
	}
	using MajorantIterator = DDAMajorantIterator;
	__device__ MajorantIterator sample_ray(const Ray& ray, float t_max, const SampledWavelengths& lambda) const;
private:
	glm::vec3 bmin, bmax;
	glm::mat4 render_from_medium;
	glm::mat4 medium_from_render;
	DenselySampledSpectrum sigma_a_spec, sigma_s_spec;
	SampledGrid<float> density_grid;
	HGPhaseFunction phase;
	SampledGrid<float> temperature_grid;
	bool isEmissive;
	DenselySampledSpectrum Le_spec;
	SampledGrid<float> Le_scale;
	MajorantGrid majorant_grid;
};

class RGBGridMedium 
{
public:
	RGBGridMedium(const glm::vec3& bmin, const glm::vec3& bmax, const glm::mat4& render_from_medium, const glm::mat4& medium_from_render, SampledGrid<RGBIlluminantSpectrum> Le_grid, float Le_scale, float g, SampledGrid<RGBUnboundedSpectrum> rgb_a, SampledGrid<RGBUnboundedSpectrum> rgb_s, float sigma_scale, Allocator alloc);
	__device__ bool is_emissive() const
	{
		return Le_grid && Le_scale > 0;
	}
	__device__ MediumProperties sample_point(const glm::vec3& point, const SampledWavelengths& lambda) const
	{
		glm::vec3 p = glm::vec3(medium_from_render * glm::vec4(point, 1.0f));
		p = (p - bmin) / (bmax - bmin);
		auto convert = [=] __host__ __device__ (RGBUnboundedSpectrum s) {return s.sample(lambda); };
		SampledSpectrum sigma_a = sigma_scale * (sigma_a_grid ? sigma_a_grid.look_up(p, convert) : SampledSpectrum(1.0f));
		SampledSpectrum sigma_s = sigma_scale * (sigma_s_grid ? sigma_s_grid.look_up(p, convert) : SampledSpectrum(1.0f));
		SampledSpectrum Le(0.f);
		if (is_emissive())
		{
			auto convert = [=] __host__ __device__ (RGBIlluminantSpectrum s) { return s.sample(lambda); };
			Le = Le_scale * Le_grid.look_up(p, convert);
		}
		return MediumProperties{ sigma_a, sigma_s, &phase, Le };
	}
	using MajorantIterator = DDAMajorantIterator;
	__device__ MajorantIterator sample_ray(const Ray& ray, float t_max, const SampledWavelengths& lambda) const;
private:
	glm::vec3 bmin, bmax;
	glm::mat4 render_from_medium;
	glm::mat4 medium_from_render;
	SampledGrid<RGBIlluminantSpectrum> Le_grid;
	float Le_scale;
	HGPhaseFunction phase;
	SampledGrid<RGBUnboundedSpectrum> sigma_a_grid, sigma_s_grid;
	float sigma_scale; 
	MajorantGrid majorant_grid;
};

class NanoVDBBuffer 
{
public:
	NanoVDBBuffer() = default;
	NanoVDBBuffer(Allocator alloc) : alloc(alloc) {}
	NanoVDBBuffer(size_t size, Allocator alloc = {}) : alloc(alloc) { init(size); }
	NanoVDBBuffer(const NanoVDBBuffer&) = delete;
	NanoVDBBuffer(NanoVDBBuffer&& other) noexcept
		: alloc(std::move(other.alloc)),
		bytesAllocated(other.bytesAllocated),
		ptr(other.ptr) {
		other.bytesAllocated = 0;
		other.ptr = nullptr;
	}
	NanoVDBBuffer& operator=(const NanoVDBBuffer&) = delete;
	NanoVDBBuffer& operator=(NanoVDBBuffer&& other) noexcept {
		// Note, this isn't how std containers work, but it's expedient for
		// our purposes here...
		clear();
		alloc = other.alloc;
		bytesAllocated = other.bytesAllocated;
		ptr = other.ptr;
		other.bytesAllocated = 0;
		other.ptr = nullptr;
		return *this;
	}
	~NanoVDBBuffer() { clear(); }

	void init(uint64_t size) {
		if (size == bytesAllocated)
			return;
		if (bytesAllocated > 0)
			clear();
		if (size == 0)
			return;
		bytesAllocated = size;
		ptr = (uint8_t*)alloc.allocate_bytes(bytesAllocated, 128);
	}

	const uint8_t* data() const { return ptr; }
	uint8_t* data() { return ptr; }
	uint64_t size() const { return bytesAllocated; }
	bool empty() const { return size() == 0; }

	void clear() {
		alloc.deallocate_bytes(ptr, bytesAllocated, 128);
		bytesAllocated = 0;
		ptr = nullptr;
	}

	static NanoVDBBuffer create(uint64_t size, const NanoVDBBuffer* context = nullptr) {
		return NanoVDBBuffer(size, context ? context->GetAllocator() : Allocator());
	}

	Allocator GetAllocator() const { return alloc; }

private:
	Allocator alloc;
	size_t bytesAllocated = 0;
	uint8_t* ptr = nullptr;
};

template <typename Buffer>
static nanovdb::GridHandle<Buffer> read_grid(const std::string& file_name, const std::string& grid_name, Allocator alloc)
{
	NanoVDBBuffer buf(alloc);
	nanovdb::GridHandle<Buffer> grid;
	try {
		grid = nanovdb::io::readGrid<Buffer>(file_name, grid_name, 1, buf);
	}
	catch (const std::exception& e) {
		std::cout << "nanovdb: " << file_name << e.what();
	}

	if (grid) {
		if (!grid.gridMetaData()->isFogVolume() && !grid.gridMetaData()->isUnknown())
			throw std::runtime_error(file_name + ":" + grid_name + " isn't a FogVolume grid?");
	}
	return grid;
}

class NanoVDBMedium 
{
public:
	using MajorantIterator = DDAMajorantIterator;
	static NanoVDBMedium* create(const BundledParams& params, const glm::mat4& render_from_medium, Allocator alloc);
	NanoVDBMedium(const glm::mat4& render_from_medium, const glm::mat4& medium_from_render, SpectrumPtr sigma_a, SpectrumPtr sigma_s, float sigma_scale, float g, nanovdb::GridHandle<NanoVDBBuffer> density_g, nanovdb::GridHandle<NanoVDBBuffer> temperature_g, float Le_scale, float temperature_offset, float temperature_scale, Allocator alloc);

	__device__ bool is_emissive() const
	{
		return temperatureFloatGrid && Le_scale > 0;
	}

	__device__ MediumProperties sample_point(const glm::vec3& point, const SampledWavelengths& lambda) const
	{
		SampledSpectrum sigma_a = sigma_a_spec.sample(lambda);
		SampledSpectrum sigma_s = sigma_s_spec.sample(lambda);

		glm::vec3 p = glm::vec3(medium_from_world * glm::vec4(point, 1.0f));

		nanovdb::Vec3<float> pIndex =
			densityFloatGrid->worldToIndexF(nanovdb::Vec3<float>(p.x, p.y, p.z));

		using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
		float d = Sampler(densityFloatGrid->tree())(pIndex);
		return MediumProperties{ sigma_a * d, sigma_s * d, &phase, Le(p, lambda) };
	}

	__device__ MajorantIterator sample_ray(const Ray& ray, float t_max, const SampledWavelengths& lambda) const;
private:
	__device__ SampledSpectrum Le(const glm::vec3& p, const SampledWavelengths& lambda) const;
	glm::vec3 bmin, bmax;
	glm::mat4 world_from_medium;
	glm::mat4 medium_from_world;
	DenselySampledSpectrum sigma_a_spec, sigma_s_spec;
	HGPhaseFunction phase;
	MajorantGrid majorant_grid;
	nanovdb::GridHandle<NanoVDBBuffer> densityGrid;
	nanovdb::GridHandle<NanoVDBBuffer> temperatureGrid;
	const nanovdb::FloatGrid* densityFloatGrid = nullptr;
	const nanovdb::FloatGrid* temperatureFloatGrid = nullptr;
	float Le_scale, temperature_offset, temperature_scale;
};


template <typename F>
using CallbackSignature = bool(const glm::vec3&, MediumProperties, SampledSpectrum, SampledSpectrum);

// F should be bool callback(const glm::vec3& p, MediumProperties mp, SampledSpectrum sigma_maj, SampledSpectrum T_maj)
template <typename F>
__device__ SampledSpectrum sample_Tmaj(MediumPtr* sceneMediumPtrs, Ray ray, float t_max, thrust::default_random_engine& rng, const SampledWavelengths& lambda, F callback)
{
	static_assert(std::is_invocable_r_v<bool, F, const glm::vec3&, MediumProperties, SampledSpectrum, SampledSpectrum>,
		"The callback must have the signature: bool(const glm::vec3&, MediumProperties, SampledSpectrum, SampledSpectrum)");
	auto sample = [&](auto medium) {
		using medium_type = typename std::remove_reference_t<decltype(*medium)>;
		return sample_Tmaj<medium_type>(medium, ray, t_max, rng, lambda, callback);
		};
	return sceneMediumPtrs[ray.medium].Dispatch(sample);
}

// F should be bool callback(const glm::vec3& p p, MediumProperties mp, SampledSpectrum sigma_maj, SampledSpectrum T_maj)
template <typename ConcreteMedium, typename F>
__device__ SampledSpectrum sample_Tmaj(MediumPtr ray_medium, Ray ray, float t_max, thrust::default_random_engine& rng, const SampledWavelengths& lambda, F callback)
{
	float ray_dir_len = glm::length(ray.direction);
	t_max *= ray_dir_len;
	ray.direction /= ray_dir_len;

	ConcreteMedium* medium = ray_medium.Cast<ConcreteMedium>();
	typename ConcreteMedium::MajorantIterator iter = medium->sample_ray(ray, t_max, lambda);

	SampledSpectrum Tmaj(1.0f);
	bool done = false;
	thrust::uniform_real_distribution<float> u01(0, 1);
	int max_iter2 = 128, iter2 = 0;
	while (!done)
	{
		float seg_tmin = 0, seg_tmax = t_max;
		SampledSpectrum sigma_maj(0.0f);
		if (!iter.next(&seg_tmin, &seg_tmax, &sigma_maj))
		{
			return Tmaj;
		}
		if (sigma_maj[0] == 0)
		{
			float dt = seg_tmax - seg_tmin;
			if (math::is_inf(dt))
			{
				dt = std::numeric_limits<float>::max();
			}
			Tmaj *= exp(-dt * sigma_maj);
			continue;
		}
		float tmin = seg_tmin;
		while (iter2 < max_iter2)
		{
			float u = u01(rng);
			float t = tmin + math::sample_exponential(u, sigma_maj[0]);
			if (t < seg_tmax)
			{
				Tmaj *= exp(-(t - tmin) * sigma_maj);
				glm::vec3 position = ray.direction * t + ray.origin;
				MediumProperties mp = medium->sample_point(position, lambda);
				if (!callback(position, mp, sigma_maj, Tmaj))
				{
					done = true;
					break;
				}
				Tmaj = SampledSpectrum(1.0f);
				tmin = t;
			}
			else
			{
				float dt = seg_tmax - tmin;
				if (math::is_inf(dt))
				{
					dt = std::numeric_limits<float>::max();
				}
				Tmaj *= exp(-dt * sigma_maj);
				break;
			}
			iter2++;
		}
	}
	return SampledSpectrum(1.0f);
}