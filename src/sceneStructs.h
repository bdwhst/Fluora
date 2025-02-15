#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "materials.h"
#include "spectrum.h"
#include "camera.h"
#include "medium.h"


#define USE_BVH 1
#define MIS_POWER_2 1
#define MTBVH 1
#define DENOISE 0
#define VIS_NORMAL 0
#define TONEMAPPING 1
#define DOF_ENABLED 1
#define SCATTER_ORIGIN_OFFSETMULT 0.0000125f
#define BOUNDING_BOX_EXPAND 0.0001f
#define ALPHA_CUTOFF 0.01f
#define STOCHASTIC_SAMPLING 1
#define FIRST_INTERSECTION_CACHING 1
#define MAX_DEPTH 32
#define SORT_BY_MATERIAL_TYPE 0
#define MAX_NUM_PRIMS_IN_LEAF 2
#define SAH_BUCKET_SIZE 20
#define SAH_RAY_BOX_INTERSECTION_COST 0.1f
#define WHITE_FURNANCE_TEST 0
#define NUM_MULTI_SCATTER_BOUNCE 16
#define WATER_TIGHT_MESH_INTERSECTION 1

enum IntegratorType {
    naive,
    mis
};



enum GeomType {
    SPHERE,
    CUBE,
    TRIANGLE_MESH
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
    int medium = -1;
};

struct ObjectTransform {
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Object {
    enum GeomType type;
    int materialid = -1;
    int meshId = -1;
    int mediumIn = -1, mediumOut = -1;
    ObjectTransform Transform;
};

struct TriangleMesh
{
    glm::vec3* m_vertices = nullptr;
    glm::vec3* m_normals = nullptr;
    glm::vec2* m_uvs = nullptr;
    glm::ivec3* m_triangles = nullptr;
};


struct BoundingBox {
    glm::vec3 pMin, pMax;
    BoundingBox() :pMin(glm::vec3(1e38f)), pMax(glm::vec3(-1e38f)) {}
    glm::vec3 center() const { return (pMin + pMax) * 0.5f; }
};

BoundingBox Union(const BoundingBox& b1, const BoundingBox& b2);
BoundingBox Union(const BoundingBox& b1, const glm::vec3& p);
float BoxArea(const BoundingBox& b);

struct Primitive {
    int objID;
    int offset;//offset for triangles in model
    int lightID;
    BoundingBox bbox;
    Primitive(const Object& obj, int objID, int triangleOffset = -1, const glm::ivec3* triangles = nullptr, const glm::vec3* vertices = nullptr);
};

struct BVHNode {
    int axis;
    BVHNode* left, * right;
    int startPrim, endPrim;
    BoundingBox bbox;
    BVHNode() :axis(-1), left(nullptr), right(nullptr), startPrim(-1), endPrim(-1) {}
};

struct BVHGPUNode
{
    int axis;
    BoundingBox bbox;
    int parent, left, right;
    int startPrim, endPrim;
    BVHGPUNode() :axis(-1), parent(-1), left(-1), right(-1), startPrim(-1), endPrim(-1){}
};


struct MTBVHGPUNode
{
    BoundingBox bbox;
    int hitLink, missLink;
    int startPrim, endPrim;
    MTBVHGPUNode():hitLink(-1), missLink(-1), startPrim(-1), endPrim(-1){}
};

const int dirs[] = {
    1,-1,2,-2,3,-3
};



//enum MaterialType {
//    diffuse, frenselSpecular, microfacet, metallicWorkflow, blinnphong, asymMicrofacet, emitting
//};

enum AsymMicrofacetType {
    conductor, dielectric
};

enum TextureType {
    color, normal, metallicroughness
};

struct GLTFTextureLoadInfo {
    char* buffer;
    int matIndex;
    TextureType texType;
    int width, height;
    int bits, component;
    GLTFTextureLoadInfo(char* buffer, int index, TextureType type, int width, int height, int bits, int component) :buffer(buffer), matIndex(index), texType(type), width(width), height(height), bits(bits), component(component){}
};

typedef glm::vec3(*phaseEvalFunc)(const glm::vec3&, const glm::vec3&, float, float, const glm::vec3&);
typedef glm::vec3(*phaseSampleFunc)(const glm::vec3&, const glm::vec3&, glm::vec3&, float, float, glm::vec3);

struct asymMicrofacetInfo
{
    AsymMicrofacetType type;
    float zs;
    float alphaXA, alphaYA;
    float alphaXB, alphaYB;
    glm::vec3 albedo;
    phaseEvalFunc fEval;
    phaseSampleFunc fSample;
};

//struct Material {
//    glm::vec3 color = glm::vec3(0);
//    float indexOfRefraction = 0.0;
//    float emittance = 0.0;
//    float metallic = -1.0;
//    float roughness = -1.0;
//    float specExponent = -1.0;
//    asymMicrofacetInfo asymmicrofacet;
//    cudaTextureObject_t baseColorMap = 0, normalMap = 0, metallicRoughnessMap = 0;
//    MaterialType type = diffuse;
//};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lensRadius = 0.001f;
    float focalLength = 1.0f;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::vector<glm::vec3> albedo;
    std::vector<glm::vec3> normal;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    SampledSpectrum transport;
    SampledSpectrum r_u, r_l;
    SampledWavelengths lambda;
    int pixelIndex;
    int depth;
    float lastMatPdf;
    bool prevSpecular;
    thrust::default_random_engine rng;
};

struct ShadowRaySegment {
    glm::vec3 pWorld;
    glm::vec3 woWorld;
    glm::vec3 normalWorld;
    SampledSpectrum transport;
    SampledSpectrum r_p;
    SampledWavelengths lambda;
    int pixelIndex;
    thrust::default_random_engine rng;
    int bsdfType = -1;
    // This is costing a bit more global memory
    char bsdfData[BxDFMaxSize];
    PhaseFunctionPtr phaseFunc = nullptr;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
    float t = FLT_MAX;
    glm::vec3 surfaceNormal = glm::vec3(0.0);
    glm::vec3 surfaceTangent = glm::vec3(0.0);
    float fsign = 1.0;
    glm::vec3 worldPos = glm::vec3(0.0);
    int materialId = -1;
    int primitiveId = -1;
    int lightId = -1;
    glm::vec2 uv = glm::vec2(0.0);
};



struct SceneInfoDev {
    MaterialPtr* dev_materials;
    MediumPtr* dev_media;
    Object* dev_objs;
    int objectsSize;
    TriangleMesh* m_dev_meshes;
    Primitive* dev_primitives;
    int m_primitives_size;
    union {
        BVHGPUNode* dev_bvhArray;
        MTBVHGPUNode* dev_mtbvhArray;
    };
    int bvhDataSize;
    cudaTextureObject_t skyboxObj;
    PixelSensor* pixelSensor;
    bool containsVolume = false;
    
};



struct SceneGbuffer {
    glm::vec3* dev_albedo;
    glm::vec3* dev_normal;
};


