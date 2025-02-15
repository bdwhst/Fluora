#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "glm/glm.hpp"
#include "memoryUtils.h"
#include "utilities.h"
#include "sceneStructs.h"
#include "bvh.h"
#include "materials.h"
#include "lights.h"


struct MaterialLoadJobInfo
{
    std::string type;
    BundledParams params;
};

struct MediumLoadJobInfo
{
    std::string type;
    BundledParams params;
    glm::mat4 world_from_medium;
};

struct TextureLoadJobInfo
{
    std::string path;
    int matId;
    std::string matTextureKey;
};

//struct LightLoadJobInfo
//{
//    std::string type;
//    BundledParams params;
//};

struct RawTextureData
{
    RawTextureData() = default;
    int width, height;
    int channelBits;
    int channels;
    std::vector<char> data;
};

struct TriangleMeshData {
    std::vector<glm::vec3> m_vertices;
    std::vector<glm::vec3> m_normals;
    std::vector<glm::vec2> m_uvs;
    std::vector<glm::ivec3> m_triangles;
};

class Scene {

private:
    std::ifstream fp_in;
    int loadMaterial(std::string materialid);
    int loadObject(std::string objectid);
    int loadCamera();
    bool loadModel(const std::string&, int, bool);
    bool loadPly(Object* newObj, const std::string& path);
    bool loadGeometry(const std::string&,int);
    void loadTextureFromFile(const std::string& texturePath, cudaTextureObject_t* texObj, RawTextureData* ret_data = nullptr);
    void LoadTextureFromMemory(void* data, int width, int height, int bits, int channels, cudaTextureObject_t* texObj);
    void loadSkybox();
    void loadJSON(const std::string&);
public:
    size_t getTriangleSize() const
    {
        size_t sz = 0;
        for (const auto& mesh : m_triangleMeshes)
        {
            sz += mesh.m_triangles.size();
        }
        return sz;
    }
    void buildBVH();
    void buildStacklessBVH();
    void LoadAllTexturesToGPU(); 
    void LoadAllMaterialsToGPU(Allocator alloc);
    void LoadAllMediaToGPU(Allocator alloc);
    void LoadAllLightsToGPU(Allocator alloc);
    void LoadAllMeshesToGPU(Allocator alloc);
    Scene(std::string filename);
    ~Scene();
    std::string sceneFilename;
    std::vector<Object> objects;
    std::vector<MaterialPtr> materials;
    std::vector<MediumPtr> media;
    std::vector<TriangleMeshData> m_triangleMeshes;
    TriangleMesh* m_dev_triangleMeshes;
    /*std::vector<glm::ivec3> triangles;
    std::vector<glm::vec3> verticies;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> m_normals;
    std::vector<glm::vec3> tangents;
    std::vector<float> fSigns;*/
    std::vector<Primitive> primitives;
    std::vector<LightPtr> lights;
    // for now we only support ONE skybox light
    LightPtr skyboxLight = nullptr;
    std::vector<BVHGPUNode> bvhArray;
    std::vector<MTBVHGPUNode> MTBVHArray;
    RenderState state;
    BVHNode* bvhroot = nullptr;
    cudaTextureObject_t skyboxTextureObj = 0;
    int bvhTreeSize = 0;
    std::vector<char*> gltfTexTmpArrays;
    std::vector<cudaArray*> textureDataPtrs;
    std::unordered_map< std::string, cudaTextureObject_t> strToTextureObj;
    std::string environmentMapPath;
    float environmentMapLuminScale = 1.0f;
    glm::vec3 environmentMapMaxLumin = glm::vec3(1e5);
    RawTextureData environmentMapData;
    std::vector<TextureLoadJobInfo> LoadTextureFromFileJobs;//texture path, materialID
    std::vector<GLTFTextureLoadInfo> LoadTextureFromMemoryJobs;
    std::vector<MaterialLoadJobInfo>  LoadMaterialJobs;
    std::vector<MediumLoadJobInfo> LoadMediumJobs;
    //std::vector<LightLoadJobInfo> LoadLightJobs;
};

struct MikkTSpaceHelper
{
    Scene* scene;
    int i;
};

struct AliasBin {
    float q, p;
    int alias = -1;
};


