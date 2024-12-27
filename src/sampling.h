#pragma once
#ifndef SAMPLING_H
#define SAMPLING_H
#include <cuda.h>
#include <glm/glm.hpp>
#include "memoryUtils.h"
#include "defines.h"
#include "mathUtils.h"

GPU_FUNC glm::vec2 util_math_uniform_sample_triangle(const glm::vec2& rand);

GPU_FUNC glm::vec3 util_sample_hemisphere_uniform(const glm::vec2& random);

GPU_FUNC glm::vec2 util_sample_disk_uniform(const glm::vec2& random);

GPU_FUNC glm::vec3 util_sample_hemisphere_cosine(const glm::vec2& random);


struct Distribution1D
{
	Distribution1D(const float* f, int n, Allocator alloc):m_func(nullptr), m_cdf(0), m_funcInt(0), m_count(n)
	{
		m_func = alloc.allocate<float>(n);
		memcpy(m_func, f, sizeof(float) * n);
		m_cdf = alloc.allocate<float>(n + 1);
		m_cdf[0] = 0;
		for (int i = 1; i < n + 1; ++i) 
			m_cdf[i] = m_cdf[i - 1] + m_func[i - 1] / n;
		m_funcInt = m_cdf[n];

		if (m_funcInt == 0) {
			for (int i = 1; i < n + 1; ++i) m_cdf[i] = float(i) / n;
		}
		else {
			for (int i = 1; i < n + 1; ++i) m_cdf[i] /= m_funcInt;
		}
	}

	CPU_GPU_FUNC float sample_continuous(float u, float* pdf, int* off = nullptr) const
	{
		int offset = math::find_interval(m_count + 1, 
			[&](int idx)
			{
				return m_cdf[idx] <= u;
			});
		if (off)
			*off = offset;
		float du = u - m_cdf[offset];
		if ((m_cdf[offset + 1] - m_cdf[offset]) > 0) {
			assert(m_cdf[offset + 1] > m_cdf[offset]);
			du /= (m_cdf[offset + 1] - m_cdf[offset]);
		}
		assert(!math::is_nan(du));
		if (pdf) *pdf = (m_funcInt > 0) ? m_func[offset] / m_funcInt : 0;

		return (offset + du) / m_count;
	}
	CPU_GPU_FUNC int sample_discrete(float u, float* pdf = nullptr, float* remapped = nullptr) const
	{
		int offset = math::find_interval(m_count + 1,
			[&](int index) 
			{ 
				return m_cdf[index] <= u; 
			});
		if (pdf) 
			*pdf = (m_funcInt > 0) ? m_func[offset] / (m_funcInt * m_count) : 0;
		if (remapped)
			*remapped = (u - m_cdf[offset]) / (m_cdf[offset + 1] - m_cdf[offset]);
		if (remapped) 
			assert(*remapped >= 0.f && *remapped <= 1.f);
		return offset;
	}

	CPU_GPU_FUNC float discrete_pdf(int index) const
	{
		assert(index >= 0 && index < m_count);
		return m_func[index] / (m_funcInt * m_count);
	}
	float* m_func;
	float* m_cdf;
	float m_funcInt;
	int m_count;
};

class Distribution2D
{
public:
	Distribution2D(const float* func, int nu, int nv, Allocator alloc) :m_pConditionalV(nullptr), m_pMarginal(nullptr), m_nu(nu), m_nv(nv)
	{
		m_pConditionalV = alloc.allocate<Distribution1D*>(nv);
		for (int v = 0; v < nv; ++v) {
			// Compute conditional sampling distribution for $\tilde{v}$
			m_pConditionalV[v] = alloc.new_object<Distribution1D>(&func[v * nu], nu, alloc);
		}
		// Compute marginal sampling distribution $p[\tilde{v}]$
		std::vector<float> marginalFunc;
		marginalFunc.reserve(nv);
		for (int v = 0; v < nv; ++v)
			marginalFunc.push_back(m_pConditionalV[v]->m_funcInt);
		m_pMarginal = alloc.new_object<Distribution1D>(&marginalFunc[0], nv, alloc);
	}

	CPU_GPU_FUNC glm::vec2 sample_continuous(const glm::vec2& u, float* pdf) const {
		float pdfs[2];
		int v;
		float d1 = m_pMarginal->sample_continuous(u[1], &pdfs[1], &v);
		float d0 = m_pConditionalV[v]->sample_continuous(u[0], &pdfs[0]);
		*pdf = pdfs[0] * pdfs[1];
		return glm::vec2(d0, d1);
	}

	CPU_GPU_FUNC float pdf(const glm::vec2& p) const {
		int iu = math::clamp(int(p[0] * m_pConditionalV[0]->m_count), 0,
			m_pConditionalV[0]->m_count - 1);
		int iv =
			math::clamp(int(p[1] * m_pMarginal->m_count), 0, m_pMarginal->m_count - 1);
		return m_pConditionalV[iv]->m_func[iu] / m_pMarginal->m_funcInt;
	}
private:
	Distribution1D** m_pConditionalV;
	Distribution1D* m_pMarginal;
	int m_nu, m_nv;
};

#endif // !SAMPLING_H
