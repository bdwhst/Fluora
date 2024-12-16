#include "media.h"
#include "mathUtils.h"
#include "intersections.h"

#include "json.hpp"

__device__ float PhaseFunctionPtr::p(const glm::vec3& wo, const glm::vec3& wi) const
{
	auto op = [&](auto ptr) { return ptr->p(wo,wi); };
	return Dispatch(op);
}

__device__ float PhaseFunctionPtr::sample_p(const glm::vec3& wo, const glm::vec2& u, glm::vec3* wi, float* pdf) const
{
	auto op = [&](auto ptr) { return ptr->sample_p(wo, u, wi, pdf); };
	return Dispatch(op);
}

__device__ float PhaseFunctionPtr::pdf(const glm::vec3& wo, const glm::vec3& wi) const
{
	auto op = [&](auto ptr) { return ptr->pdf(wo, wi); };
	return Dispatch(op);
}

MediumPtr MediumPtr::create(const std::string& type, const BundledParams& params, const glm::mat4& world_from_medium, Allocator alloc)
{
	if (type == "nanovdb")
	{
		return NanoVDBMedium::create(params, world_from_medium, alloc);
	}
	else if (type == "homogeneous")
	{
		return HomogeneousMedium::create(params, alloc);
	}
	else
	{
		assert(0);
		return nullptr;
	}
}

__device__ bool MediumPtr::is_emissive() const
{
	auto op = [&](auto ptr) { return ptr->is_emissive(); };
	return Dispatch(op);
}

__device__ MediumProperties MediumPtr::sample_point(const glm::vec3& point, const SampledWavelengths& lambda) const
{
	auto op = [&](auto ptr) { return ptr->sample_point(point, lambda); };
	return Dispatch(op);
}




static __device__ float henyey_greenstein(float cosTheta, float g)
{
	float denom = 1 + math::sqr(g) + 2 * g * cosTheta;
	return 1 / (4.0f * math::pi) * (1 - math::sqr(g)) / (denom * sqrtf(math::max(denom,0)));
}

static __device__ glm::vec3 sample_henyey_greenstein(const glm::vec3& wo, float g, const glm::vec2& u, float* pdf)
{
	using namespace math;
	float cosTheta;
	if (abs(g) < 1e-3f)
	{
		cosTheta = 1 - 2 * u[0];
	}
	else
	{
		cosTheta = -1 / (2 * g) * (1 + sqr(g) - sqr((1 - sqr(g)) / (1 + g - 2 * g * u[0])));
	}
	float sinTheta = safe_sqrt(1 - sqr(cosTheta));
	float phi = 2 * pi * u[1];
	Frame frame = Frame::from_z(wo);
	glm::vec3 wi = glm::normalize(frame.from_local(spherical_direction(sinTheta, cosTheta, phi)));
	if (pdf) *pdf = henyey_greenstein(cosTheta, g);
	return wi;
}

__device__ float HGPhaseFunction::p(const glm::vec3& wo, const glm::vec3& wi) const
{
	return henyey_greenstein(glm::dot(wo, wi), g);
}

__device__ float HGPhaseFunction::sample_p(const glm::vec3& wo, const glm::vec2& u, glm::vec3* wi, float* pdf) const
{
	*wi = sample_henyey_greenstein(wo, g, u, pdf);
	return *pdf;
}

__device__ float HGPhaseFunction::pdf(const glm::vec3& wo, const glm::vec3& wi) const
{
	return p(wo, wi);
}

HomogeneousMedium* HomogeneousMedium::create(const BundledParams& params, Allocator alloc)
{
	SpectrumPtr Le = params.get_spec("Le");
	float Le_scale = params.get_float("Lescale", 1.0f);
	SpectrumPtr sigma_a = params.get_spec("sigma_a");
	if (!sigma_a)
		sigma_a = alloc.new_object<ConstantSpectrum>(1.0f);
	SpectrumPtr sigma_s = params.get_spec("sigma_s");
	if (!sigma_s)
		sigma_s = alloc.new_object<ConstantSpectrum>(1.0f);
	float sigma_scale = params.get_float("scale", 1.0f);
	float g = params.get_float("g", 0.0f);
	return alloc.new_object<HomogeneousMedium>(sigma_a, sigma_s, sigma_scale, Le, Le_scale, g, alloc);
}


GridMedium::GridMedium(const glm::vec3& bmin, const glm::vec3& bmax, const glm::mat4& render_from_medium, const glm::mat4& medium_from_render, SpectrumPtr sigma_a, SpectrumPtr sigma_s, float sigma_scale, float g, SampledGrid<float> d, SampledGrid<float> temperature, SpectrumPtr Le, SampledGrid<float> LeGrid, Allocator alloc) :
	bmin(bmin), bmax(bmax), render_from_medium(render_from_medium), medium_from_render(medium_from_render), sigma_a_spec(sigma_a, alloc), sigma_s_spec(sigma_s, alloc), density_grid(d), phase(g), temperature_grid(temperature), Le_spec(Le, alloc), Le_scale(LeGrid), majorant_grid(bmin, bmax, { 16,16,16 }, alloc)
{
	//TODO: scale sigma_a_spec sigma_s_spec
	isEmissive = temperature_grid ? true : (Le_spec.max_value() > 0);
	for (int z = 0; z < majorant_grid.res.z; ++z)
		for (int y = 0; y < majorant_grid.res.y; ++y)
			for (int x = 0; x < majorant_grid.res.x; ++x) {
				glm::vec3 t_bmin, t_bmax;
				majorant_grid.get_voxel_bounds(x, y, z, &t_bmin, &t_bmax);
				majorant_grid.set(x, y, z, density_grid.max_value(t_bmin, t_bmax));
			}
}

__device__ GridMedium::MajorantIterator GridMedium::sample_ray(const Ray& ray, float t_max, const SampledWavelengths& lambda) const
{
	// TODO: move this to a function
	glm::vec3 ori = glm::vec3(medium_from_render * glm::vec4(ray.origin, 1.0f));
	glm::vec3 dir = glm::vec3(medium_from_render * glm::vec4(ray.direction, 0.0f));
	float tt_min, tt_max;
	// ?? here the tt_min and tt_max are for transformed ray, do we need to account for it when we are computing the transmittance ??
	if (!util_geometry_ray_box_intersection(bmin, bmax, { ori, dir }, t_max, &tt_min, &tt_max))
	{
		return {};
	}
	SampledSpectrum sigma_a = sigma_a_spec.sample(lambda);
	SampledSpectrum sigma_s = sigma_s_spec.sample(lambda);

	SampledSpectrum sigma_t = sigma_a + sigma_s;
	DDAMajorantIterator tmp({ ori, dir }, tt_min, tt_max, &majorant_grid, sigma_t);
	return tmp;
}

RGBGridMedium::RGBGridMedium(const glm::vec3& bmin, const glm::vec3& bmax, const glm::mat4& render_from_medium, const glm::mat4& medium_from_render, SampledGrid<RGBIlluminantSpectrum> Le_grid, float Le_scale, float g, SampledGrid<RGBUnboundedSpectrum> rgb_a, SampledGrid<RGBUnboundedSpectrum> rgb_s, float sigma_scale, Allocator alloc) :
	bmin(bmin), bmax(bmax), render_from_medium(render_from_medium), medium_from_render(medium_from_render), Le_grid(Le_grid), Le_scale(Le_scale), phase(g), sigma_a_grid(rgb_a), sigma_s_grid(rgb_s), sigma_scale(sigma_scale), majorant_grid(bmin,bmax, {16,16,16}, alloc)
{
	for (int z = 0; z < majorant_grid.res.z; ++z)
		for (int y = 0; y < majorant_grid.res.y; ++y)
			for (int x = 0; x < majorant_grid.res.x; ++x) {
				glm::vec3 t_bmin, t_bmax;
				majorant_grid.get_voxel_bounds(x, y, z, &t_bmin, &t_bmax);
				auto max_func = [] (RGBUnboundedSpectrum s) {
					return s.max_value();
				};
				float maxSigma_t =
					(sigma_a_grid ? sigma_a_grid.max_value(t_bmin, t_bmax, max_func) : 1) +
					(sigma_s_grid ? sigma_s_grid.max_value(t_bmin, t_bmax, max_func) : 1);
				majorant_grid.set(x, y, z, sigma_scale* maxSigma_t);
			}
}

__device__ RGBGridMedium::MajorantIterator RGBGridMedium::sample_ray(const Ray& ray, float t_max, const SampledWavelengths& lambda) const
{
	// TODO: move this to a function
	glm::vec3 ori = glm::vec3(medium_from_render * glm::vec4(ray.origin, 1.0f));
	glm::vec3 dir = glm::vec3(medium_from_render * glm::vec4(ray.direction, 0.0f));
	float tt_min, tt_max;
	// ?? here the tt_min and tt_max are for transformed ray, do we need to account for it when we are computing the transmittance ??
	if (!util_geometry_ray_box_intersection(bmin, bmax, { ori, dir }, t_max, &tt_min, &tt_max))
	{
		return {};
	}
	SampledSpectrum sigma_t(1.0f);
	DDAMajorantIterator tmp({ ori, dir }, tt_min, tt_max, &majorant_grid, sigma_t);
	return tmp;
}

NanoVDBMedium* NanoVDBMedium::create(const BundledParams& params, const glm::mat4& world_from_medium, Allocator alloc)
{
	std::string filename = params.get_string("filename");
	if (filename.empty())
		throw std::runtime_error("filename is not found in params");

	nanovdb::GridHandle<NanoVDBBuffer> density_grid;
	std::string gridname = params.get_string("gridname", "density");
	if (gridname.empty())
		throw std::runtime_error("gridname is not found in params");

	density_grid = read_grid<NanoVDBBuffer>(filename, gridname, alloc);
	if (!density_grid)
		throw std::runtime_error(filename + ": didn't find \"density\" grid.");

	// optional
	nanovdb::GridHandle<NanoVDBBuffer> temperature_grid;
	std::string temperaturename =
		params.get_string("temperaturename", "temperature");
	temperature_grid = read_grid<NanoVDBBuffer>(filename, temperaturename, alloc);

	float Le_scale = params.get_float("Lescale", 1.0f);
	float temperatureOffset = params.get_float("temperatureoffset",
		params.get_float("temperaturecutoff", 0.f));
	float temperatureScale = params.get_float("temperaturescale", 1.f);
	float g = params.get_float("g", 0.);

	SpectrumPtr sigma_a = params.get_spec("sigma_a");
	if (!sigma_a)
		sigma_a = alloc.new_object<ConstantSpectrum>(1.f);
	SpectrumPtr sigma_s = params.get_spec("sigma_s");
	if (!sigma_s)
		sigma_s = alloc.new_object<ConstantSpectrum>(1.f);
	float sigma_scale = params.get_float("scale", 1.f);

	return alloc.new_object<NanoVDBMedium>(world_from_medium, glm::inverse(world_from_medium), sigma_a, sigma_s, sigma_scale, g, std::move(density_grid), std::move(temperature_grid), Le_scale, temperatureOffset, temperatureScale, alloc);
}


NanoVDBMedium::NanoVDBMedium(const glm::mat4& world_from_medium, const glm::mat4& medium_from_world, SpectrumPtr sigma_a, SpectrumPtr sigma_s, float sigma_scale, float g, nanovdb::GridHandle<NanoVDBBuffer> density_g, nanovdb::GridHandle<NanoVDBBuffer> temperature_g, float Le_scale, float temperature_offset, float temperature_scale, Allocator alloc):
	world_from_medium(world_from_medium), medium_from_world(medium_from_world), 
	sigma_a_spec(sigma_a, alloc), sigma_s_spec(sigma_s, alloc), 
	phase(g), majorant_grid(glm::vec3(), glm::vec3(),{64,64,64},alloc), 
	densityGrid(std::move(density_g)), temperatureGrid(std::move(temperature_g)), 
	Le_scale(Le_scale), temperature_offset(temperature_offset), temperature_scale(temperature_scale)
{
	densityFloatGrid = densityGrid.grid<float>();

	sigma_a_spec.scale(sigma_scale);
	sigma_s_spec.scale(sigma_scale);

	nanovdb::BBox<nanovdb::Vec3R> bbox = densityFloatGrid->worldBBox();
	bmin = glm::vec3(bbox.min()[0], bbox.min()[1], bbox.min()[2]);
	bmax = glm::vec3(bbox.max()[0], bbox.max()[1], bbox.max()[2]);

	if (temperatureGrid)
	{
		temperatureFloatGrid = temperatureGrid.grid<float>();
		float minTemperature, maxTemperature;
		temperatureFloatGrid->tree().extrema(minTemperature, maxTemperature);
		nanovdb::BBox<nanovdb::Vec3R> bbox = temperatureFloatGrid->worldBBox();
		bmin = glm::min(bmin, glm::vec3(bbox.min()[0], bbox.min()[1], bbox.min()[2]));
		bmax = glm::max(bmax, glm::vec3(bbox.max()[0], bbox.max()[1], bbox.max()[2]));
	}
	majorant_grid.bmin = bmin;
	majorant_grid.bmax = bmax;
	// TODO: make this a parallel op
	for (int z = 0; z < majorant_grid.res.z; z++)
	{
		for (int y = 0; y < majorant_grid.res.y; y++)
		{
			for (int x = 0; x < majorant_grid.res.x; x++)
			{
				glm::vec3 world_bmin = math::lerp(glm::vec3((float)x / majorant_grid.res.x, (float)y / majorant_grid.res.y, (float)z / majorant_grid.res.z), bmin, bmax);
				glm::vec3 world_bmax = math::lerp(glm::vec3((float)(x + 1) / majorant_grid.res.x, (float)(y + 1) / majorant_grid.res.y, (float)(z + 1) / majorant_grid.res.z), bmin, bmax);

				nanovdb::Vec3R i0 = densityFloatGrid->worldToIndexF(
					nanovdb::Vec3R(world_bmin.x, world_bmin.y, world_bmin.z));
				nanovdb::Vec3R i1 = densityFloatGrid->worldToIndexF(
					nanovdb::Vec3R(world_bmax.x, world_bmax.y, world_bmax.z));

				auto bbox = densityFloatGrid->indexBBox();
				float delta = 1.f;  // Filter slop
				int nx0 = math::max(int(i0[0] - delta), bbox.min()[0]);
				int nx1 = math::min(int(i1[0] + delta), bbox.max()[0]);
				int ny0 = math::max(int(i0[1] - delta), bbox.min()[1]);
				int ny1 = math::min(int(i1[1] + delta), bbox.max()[1]);
				int nz0 = math::max(int(i0[2] - delta), bbox.min()[2]);
				int nz1 = math::min(int(i1[2] + delta), bbox.max()[2]);

				// FIXME: While the following is properly conservative, it can lead
				// to voxels with majorants that are much higher than any actual
				// volume density value in their extent. The issue comes up when
				// a) the density at a sample outside the is much higher than in the
				// voxel's interior and b) that sample only has a minimal
				// contribution in practice due to trilinear interpolation.  We
				// compute a majorant as if it might fully contribute, even though
				// it can't.  Fixing this would require careful handling of the
				// boundary samples.  The impact of these majorants is not
				// insignificant; they cause a roughly 10% slowdown in practice
				// due to excess null scattering in such voxels.
				float maxValue = 0;
				auto accessor = densityFloatGrid->getAccessor();

				// nanovdb integer bounding boxes are inclusive on the upper end...
				for (int nz = nz0; nz <= nz1; ++nz)
					for (int ny = ny0; ny <= ny1; ++ny)
						for (int nx = nx0; nx <= nx1; ++nx)
							maxValue = std::max(maxValue, accessor.getValue({ nx, ny, nz }));

				majorant_grid.set(x, y, z, maxValue);
			}
		}
	}
#if 0
	{
		using json = nlohmann::json;
		std::string filename = "grid.json";
		json gridData;
		gridData["res"] = { majorant_grid.res.x,majorant_grid.res.y,majorant_grid.res.z };
		gridData["data"] = json::array();
		for (int i = 0; i < majorant_grid.res.x * majorant_grid.res.y * majorant_grid.res.z; i++)
		{
			gridData["data"].emplace_back(majorant_grid.voxels[i]);
		}
		std::ofstream out(filename);
		if (out.is_open())
		{
			out << gridData.dump();
			out.close();
		}
		else
		{
			throw std::runtime_error(filename + " cannot be open");
		}
	}
#endif
}

__device__ NanoVDBMedium::MajorantIterator NanoVDBMedium::sample_ray(const Ray& ray, float t_max, const SampledWavelengths& lambda) const
{
	// TODO: move this to a function
	glm::vec3 ori = glm::vec3(medium_from_world * glm::vec4(ray.origin, 1.0f));
	glm::vec3 dir = glm::vec3(medium_from_world * glm::vec4(ray.direction, 0.0f));
	float tt_min, tt_max;
	// ?? here the tt_min and tt_max are for transformed ray, do we need to account for it when we are computing the transmittance ??
	if (!util_geometry_ray_box_intersection(bmin, bmax, { ori, dir }, t_max, &tt_min, &tt_max))
	{
		return {};
	}
	assert(tt_max <= t_max);
	SampledSpectrum sigma_a = sigma_a_spec.sample(lambda);
	SampledSpectrum sigma_s = sigma_s_spec.sample(lambda);

	SampledSpectrum sigma_t = sigma_a + sigma_s;
	return DDAMajorantIterator({ori,dir,ray.medium}, tt_min, tt_max, &majorant_grid, sigma_t);
}

__device__ SampledSpectrum NanoVDBMedium::Le(const glm::vec3& p, const SampledWavelengths& lambda) const {
	if (!temperatureFloatGrid)
		return SampledSpectrum(0.f);
	nanovdb::Vec3<float> pIndex =
		temperatureFloatGrid->worldToIndexF(nanovdb::Vec3<float>(p.x, p.y, p.z));
	using Sampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
	float temp = Sampler(temperatureFloatGrid->tree())(pIndex);
	temp = (temp - temperature_offset) * temperature_scale;
	if (temp <= 100.f)
		return SampledSpectrum(0.f);
	return Le_scale * BlackbodySpectrum(temp).sample(lambda);
}

