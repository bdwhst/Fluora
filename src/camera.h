#pragma once

#include "glm/glm.hpp"
#include "spectrum.h"
#include "color.h"
class PixelSensor {
public:
	PixelSensor(SpectrumPtr r, SpectrumPtr g, SpectrumPtr b, const RGBColorSpace* outputSpace, SpectrumPtr sensorIllum, float imagingRatio, Allocator alloc);
	PixelSensor(const RGBColorSpace* outputSpace, SpectrumPtr sensorIllum, float imagingRatio, Allocator alloc);
	__device__ glm::vec3 to_sensor_rgb(SampledSpectrum L, const SampledWavelengths& lambda)
	{
		L = safe_div(L, lambda.pdf());
		return imagingRatio * glm::vec3((r_bar.sample(lambda) * L).average(), (g_bar.sample(lambda) * L).average(), (b_bar.sample(lambda) * L).average());
	}

	glm::mat3 XYZFromSensorRGB;
private:
	glm::vec3 project_reflectance(SpectrumPtr refl, SpectrumPtr illum, SpectrumPtr b1, SpectrumPtr b2, SpectrumPtr b3);
	static constexpr int nSwatchReflectances = 24;
	static SpectrumPtr swatchReflectances[nSwatchReflectances];
	DenselySampledSpectrum r_bar, g_bar, b_bar;
	float imagingRatio;

};


class RGBFilm
{
public:
	RGBFilm(glm::vec3* image, const RGBColorSpace* outputSpace, float threshold):image(image),outputSpace(outputSpace),threshold(threshold) {}

	__device__ void add_radiance(const glm::vec3& sensorRGB, int pixelIndex);
	__device__ void add_rgb(const glm::vec3& rgb, int pixelIndex);
	__device__ const glm::vec3* get_image() const
	{
		return image;
	}
private:
	glm::vec3* image;
	const RGBColorSpace* outputSpace;
	float threshold;
};