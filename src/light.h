#pragma once

#include "taggedptr.h"
#include "spectrum.h"
#include "sceneStructs.h"
#include "memoryUtils.h"
#include <unordered_set>

enum class LightType
{
    DeltaPosition,
    DeltaDirection,
    Area,
    Infinite
};

struct LightSampleContext {
    glm::vec3 pi,n,ns;
    Primitive* dev_primitives;
    Object* dev_objects;
    ModelInfoDev modelInfo;
};

struct LightLiSample {
    SampledSpectrum L;
    glm::vec3 wi;
    float pdf;
    glm::vec3 pLight;
};

class DiffuseAreaLight;

class LightPtr : public TaggedPointer<DiffuseAreaLight>
{
public:
    using TaggedPointer::TaggedPointer;
    __device__ SampledSpectrum phi(SampledWavelengths lambda) const;
    __device__ LightType type() const;
    __device__ bool is_delta_light(LightType type) const
    {
        return (type == LightType::DeltaPosition || type == LightType::DeltaDirection);
    }
    __device__ LightLiSample sample_Li(const LightSampleContext& ctx, glm::vec3 rand, SampledWavelengths lambda, bool allowIncompletePDF = false) const;
    __device__ float pdf_Li(const LightSampleContext& ctx, const glm::vec3& pLight, const glm::vec3& nLight, bool allowIncompletePDF = false);
    __device__ SampledSpectrum L(const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv, const glm::vec3& w, const SampledWavelengths& lambda) const;
    __device__ SampledSpectrum Le(const Ray& ray, const SampledWavelengths& lambda) const;
};

// Hash set for instance's ptr
template<typename T, typename Hash, typename Equal>
class InstanceSet
{
public:
    InstanceSet(Allocator alloc) :m_alloc(alloc) {}
    const T* lookup(const T& item)
    {
        auto iter = m_data.find(&item);
        if (iter != m_data.end())
        {
            return *iter;
        }
        T* ptr = m_alloc.new_object<T>(item);
        m_data.insert(ptr);
        return ptr;
    }
    template<typename F>
    const T* lookup(const T& item, F create)
    {
        auto iter = m_data.find(&item);
        if (iter != m_data.end())
        {
            return *iter;
        }
        T* ptr = create(m_alloc, item);
        m_data.insert(ptr);
        return ptr;
    }
private:
    Allocator m_alloc;
    std::unordered_set<const T*, Hash, Equal> m_data;
};

class LightBase
{
public:
    LightBase(LightType type):m_type(type){}
    __device__ LightType type() const { return m_type; }
    __device__ SampledSpectrum L(const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv, const glm::vec3& w, const SampledWavelengths& lambda) const { return SampledSpectrum(0.0f); }
    __device__ SampledSpectrum Le(const Ray& ray, const SampledWavelengths& lambda) const { return SampledSpectrum(0.0f); }
    using SpectrumInstanceSet = InstanceSet<DenselySampledSpectrum, DenselySampledSpectrumPtrHash, DenselySampledSpectrumPtrEqual>;
protected:
    static const DenselySampledSpectrum* lookup_spectrum(SpectrumPtr s)
    {
        if (m_spectrumInstanceSet == nullptr)
            m_spectrumInstanceSet = new SpectrumInstanceSet(Allocator(CUDAMemoryResourceBackend::getInstance()));
        // TODO: recycle memory here
        return m_spectrumInstanceSet->lookup(DenselySampledSpectrum(s, {}), [&](Allocator alloc, const DenselySampledSpectrum& spec) {
            return alloc.new_object<DenselySampledSpectrum>(&spec, alloc);
        });
    }
    LightType m_type;
    static SpectrumInstanceSet* m_spectrumInstanceSet;
};