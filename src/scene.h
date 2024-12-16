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

class Scene {

private:
    std::ifstream fp_in;
    int loadMaterial(std::string materialid);
    int loadObject(std::string objectid);
    int loadCamera();
    bool loadModel(const std::string&, int, bool);
    bool loadGeometry(const std::string&,int);
    void loadTextureFromFile(const std::string& texturePath, cudaTextureObject_t* texObj, int type);
    void LoadTextureFromMemory(void* data, int width, int height, int bits, int channels, cudaTextureObject_t* texObj);
    void loadSkybox();
    void loadJSON(const std::string&);
public:
    void buildBVH();
    void buildStacklessBVH();
    void LoadAllTexturesToGPU(); 
    void LoadAllMaterialsToGPU(Allocator alloc);
    void LoadAllMediaToGPU(Allocator alloc);
    void CreateLights();
    Scene(std::string filename);
    ~Scene();

    std::vector<Object> objects;
    std::vector<MaterialPtr> materials;
    std::vector<MediumPtr> media;
    std::vector<glm::ivec3> triangles;
    std::vector<glm::vec3> verticies;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> tangents;
    std::vector<float> fSigns;
    std::vector<Primitive> primitives;
    std::vector<Primitive> lights;
    std::vector<BVHGPUNode> bvhArray;
    std::vector<MTBVHGPUNode> MTBVHArray;
    RenderState state;
    BVHNode* bvhroot = nullptr;
    cudaTextureObject_t skyboxTextureObj = 0;
    int bvhTreeSize = 0;
    std::vector<char*> gltfTexTmpArrays;
    std::vector<cudaArray*> textureDataPtrs;
    std::unordered_map< std::string, cudaTextureObject_t> strToTextureObj;
    std::vector<std::pair<std::string, int> > LoadTextureFromFileJobs;//texture path, materialID
    std::vector<GLTFTextureLoadInfo> LoadTextureFromMemoryJobs;
    std::vector<MaterialLoadJobInfo>  LoadMaterialJobs;
    std::vector<MediumLoadJobInfo> LoadMediumJobs;
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


