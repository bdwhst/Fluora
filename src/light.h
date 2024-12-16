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
};

struct LightLiSample {
    SampledSpectrum L;
    glm::vec3 wi;
    float pdf;
    glm::vec3 pLight;
};

class DiffuseAreaLight;

class Light : public TaggedPointer<DiffuseAreaLight>
{
public:
    using TaggedPointer::TaggedPointer;
    SampledSpectrum phi(SampledWavelengths lambda) const;
    LightType type() const;
    bool is_delta_light(LightType type) const
    {
        return (type == LightType::DeltaPosition || type == LightType::DeltaDirection);
    }
    LightLiSample sample_Li(const LightSampleContext& ctx, glm::vec2 rand, SampledWavelengths lambda, bool allowIncompletePDF = false) const;
    float pdf_Li(const LightSampleContext& ctx, const glm::vec3& wi, bool allowIncompletePDF = false);
    SampledSpectrum L(const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv, const glm::vec3& w, const SampledWavelengths& lambda) const;
    SampledSpectrum Le(const Ray& ray, const SampledWavelengths& lambda) const;


};

template<typename T, typename Hash = std::hash<T>>
class InstanceSet
{
public:
    InstanceSet(Allocator alloc) :mAlloc(alloc) {}
    const T* lookup(const T& item)
    {
        auto iter = mData.find(&item);
        if (iter != mData.end())
        {
            return *iter;
        }
        T* ptr = mAlloc.new_object<T>(item);
        mData.insert(ptr);
        return ptr;
    }
    template<typename F>
    const T* lookup(const T& item, F create)
    {
        auto iter = mData.find(&item);
        if (iter != mData.end())
        {
            return *iter;
        }
        T* ptr = create(mAlloc, item);
        mData.insert(ptr);
        return ptr;
    }
private:
    Allocator mAlloc;
    std::unordered_set<T*, Hash> mData;
};

class LightBase
{
public:
    LightBase(LightType type, const glm::mat4& renderFromLight);
    LightType type() const { return mType; }
    SampledSpectrum L(const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv, const glm::vec3& w, const SampledWavelengths& lambda) const { return SampledSpectrum(0.0f); }
    SampledSpectrum Le(const Ray& ray, const SampledWavelengths& lambda) const { return SampledSpectrum(0.0f); }
protected:
    static const DenselySampledSpectrum* lookup_spectrum(SpectrumPtr s)
    {
        if (mSpectrumInstanceSet == nullptr)
            mSpectrumInstanceSet = new InstanceSet<DenselySampledSpectrum>(Allocator(CUDAMemoryResourceBackend::getInstance()));
        return mSpectrumInstanceSet->lookup(DenselySampledSpectrum(s, Allocator(CUDAMemoryResourceBackend::getInstance())));
    }
    LightType mType;
    glm::mat4 mRenderFromLight;
    static InstanceSet<DenselySampledSpectrum>* mSpectrumInstanceSet;
};