#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp >
#include <glm/gtx/string_cast.hpp>
#include <stb_image.h>
#include <stb_image_write.h>
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#define TINYPLY_IMPLEMENTATION
#include <tinyply.h>
#define TINYEXR_USE_MINIZ 0 
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>
#include <mikktspace.h>
#include <unordered_map>
#include <queue>
#include <numeric>
#include <json.hpp>

#include "memoryUtils.h"
#include "materials.h"
#include "scene.h"

Scene::Scene(std::string filename):sceneFilename(filename) {
    using namespace std;
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    string ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".txt")
    {
        char* fname = (char*)filename.c_str();
        fp_in.open(fname);
        if (!fp_in.is_open()) {
            cout << "Error reading from file - aborting!" << endl;
            throw;
        }
        while (fp_in.good()) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            if (!line.empty()) {
                vector<string> tokens = utilityCore::tokenizeString(line);
                if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                    loadMaterial(tokens[1]);
                    cout << " " << endl;
                }
                else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                    loadObject(tokens[1]);
                    cout << " " << endl;
                }
                else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                    loadCamera();
                    cout << " " << endl;
                }
                else if (strcmp(tokens[0].c_str(), "SKYBOX") == 0) {
                    loadSkybox();
                    cout << " " << endl;
                }
            }
        }
    }
    else if (ext == ".json")
    {
        loadJSON(filename);
    }
}

void Scene::loadJSON(const std::string& name)
{
    using json = nlohmann::json;
    std::ifstream f(name);
    json data = json::parse(f);

    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    if (cameraData.contains("LENS_RADIUS"))
    {
        camera.lensRadius = cameraData["LENS_RADIUS"];
    }
    if (cameraData.contains("FOCAL_LEN"))
    {
        camera.focalLength = cameraData["FOCAL_LEN"];
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());


    const auto& backgroundData = data["Background"];
    if (backgroundData["TYPE"] == "skybox")
        environmentMapPath = backgroundData["PATH"];
    if (backgroundData.contains("SCALE"))
    {
        environmentMapLuminScale = backgroundData["SCALE"];
    }
    else
    {
        environmentMapLuminScale = 1.0f;
    }
    if (backgroundData.contains("MAXRGB"))
    {
        environmentMapMaxLumin = glm::vec3(backgroundData["MAXRGB"][0], backgroundData["MAXRGB"][1], backgroundData["MAXRGB"][2]);
    }

    // TODO: load materials here
    std::unordered_map<std::string, int> materialNameToId;
    if (data.contains("Materials"))
    {
        int currMatId = LoadMaterialJobs.size();
        for (const auto& [key, val] : data["Materials"].items())
        {
            BundledParams params;
            std::string type = val["TYPE"];
            if (type == "diffuse")
            {
                auto& refl = val["REFL"];
                if (refl["TYPE"] == "RGB")
                {
                    params.insert_vec3("albedo", glm::vec3(refl["VALUE"][0], refl["VALUE"][1], refl["VALUE"][2]));
                }
                else if (refl["TYPE"] == "TEX")
                {
                    LoadTextureFromFileJobs.emplace_back(refl["VALUE"], currMatId, "albedoMap");
                }
            }
            else if (type == "emissive")
            {
                float emittance = 1.0f;
                if (val.contains("EMITTANCE"))
                    emittance = val["EMITTANCE"];
                params.insert_float("emittance", emittance);
                params.insert_vec3("albedo", glm::vec3(val["RGB"][0], val["RGB"][1], val["RGB"][2]));
            }
            else if (type == "dielectric")
            {
                auto& eta = val["ETA"];
                if (eta["TYPE"] == "const")
                {
                    params.insert_float("eta", eta["VALUE"]);
                }
                // TODO
            }
            else if (type == "conductor")
            {
                auto& eta = val["ETA"];
                auto& k = val["K"];
                if(eta["TYPE"]=="named")
                    params.insert_string("eta", eta["VALUE"]);
                if (k["TYPE"] == "named")
                    params.insert_string("k", k["VALUE"]);
                params.insert_float("roughness", val["ROUGHNESS"]);
            }
            if (val.contains("NORMAL_MAP"))
            {
                LoadTextureFromFileJobs.emplace_back(val["NORMAL_MAP"], currMatId, "normalMap");
            }
            int id = (int)LoadMaterialJobs.size();
            materialNameToId[key] = id;
            LoadMaterialJobs.emplace_back(type, params);
            currMatId++;
        }
    }

    std::unordered_map <std::string, std::pair<std::string, std::string>> mediumInterfaces;
    for (const auto& [key, val] : data["MediumInterfaces"].items())
    {
        mediumInterfaces[key] = std::make_pair(val["INSIDE"], val["OUTSIDE"]);
    }

    std::unordered_map<std::string, int> mediaNameToID;
    for (const auto& [key, val] : data["Media"].items())
    {
        BundledParams params;
        const std::string& type = val["TYPE"];
        if (type == "nanovdb")
        {
            params.insert_string("filename", val["PATH"]);
            if(val.contains("TEMPSCALE"))
                params.insert_float("temperaturescale", val["TEMPSCALE"]);
            if (val.contains("TEMPOFFSET"))
                params.insert_float("temperatureoffset", val["TEMPOFFSET"]);
        }
        if (val.contains("LESCALE"))
            params.insert_float("Lescale", val["LESCALE"]);
        
        // load rgb here, create spectrum later
        params.insert_vec3("sigma_a_rgb", { val["SIGMA_A"]["VALUE"][0], val["SIGMA_A"]["VALUE"][1], val["SIGMA_A"]["VALUE"][2]});
        params.insert_vec3("sigma_s_rgb", { val["SIGMA_S"]["VALUE"][0], val["SIGMA_S"]["VALUE"][1], val["SIGMA_S"]["VALUE"][2]});
        if(val.contains("SIGMA_SCALE"))
            params.insert_float("scale", val["SIGMA_SCALE"]);

        if (val.contains("G"))
        {
            params.insert_float("g", val["G"]);
        }

        glm::mat4 world_from_medium;
        if (val.contains("TRANS") || val.contains("ROTAT") || val.contains("SCALE"))
        {
            ObjectTransform modelTrans;
            modelTrans.translation = glm::vec3(val["TRANS"][0], val["TRANS"][1], val["TRANS"][2]);
            modelTrans.rotation = glm::vec3(val["ROTAT"][0], val["ROTAT"][1], val["ROTAT"][2]);
            modelTrans.scale = glm::vec3(val["SCALE"][0], val["SCALE"][1], val["SCALE"][2]);

            modelTrans.transform = utilityCore::buildTransformationMatrix(
                modelTrans.translation, modelTrans.rotation, modelTrans.scale);
            world_from_medium = modelTrans.transform;
        }

        mediaNameToID[key] = LoadMediumJobs.size();
        LoadMediumJobs.emplace_back(type, params, world_from_medium);
    }

    for (const auto& ele : data["Objects"])
    {
        const std::string type = ele["TYPE"];
        Object new_obj;
        ObjectTransform modelTrans;
        if (type == "model_inline")
        {
            new_obj.type = TRIANGLE_MESH;
            TriangleMeshData newMesh;
            for (int i = 0; i < ele["VERTICES"].size(); i += 3)
            {
                glm::vec3 tmp_pos(ele["VERTICES"][i], ele["VERTICES"][i + 1], ele["VERTICES"][i + 2]);
                newMesh.m_vertices.emplace_back(tmp_pos);
            }
            for (int i = 0; i < ele["INDICES"].size(); i += 3)
            {
                glm::ivec3 tmp_idx(ele["INDICES"][i], ele["INDICES"][i + 1], ele["INDICES"][i + 2]);
                newMesh.m_triangles.emplace_back(tmp_idx);
            }
            

            if (ele.contains("NORMALS"))
            {
                // TODO
                assert(0);
            }
            new_obj.meshId = static_cast<int>(m_triangleMeshes.size());
            m_triangleMeshes.emplace_back(std::move(newMesh));
        }
        else if (type == "model_ply")
        {
            loadPly(&new_obj,ele["PATH"]);
        }
        else if (type == "geometry_cube")
        {
            new_obj.type = CUBE;
        }

        if (ele.contains("MEDIUM_INTERFACE"))
        {
            const auto& interface = mediumInterfaces[ele["MEDIUM_INTERFACE"]];

            new_obj.mediumIn = mediaNameToID.contains(interface.first) ? mediaNameToID[interface.first] : -1;
            new_obj.mediumOut = mediaNameToID.contains(interface.second) ? mediaNameToID[interface.second] : -1;
        }

        if (ele.contains("MATERIAL"))
        {
            new_obj.materialid = materialNameToId[ele["MATERIAL"]];
        }

        modelTrans.translation = glm::vec3(ele["TRANS"][0], ele["TRANS"][1], ele["TRANS"][2]);
        modelTrans.rotation = glm::vec3(ele["ROTAT"][0], ele["ROTAT"][1], ele["ROTAT"][2]);
        modelTrans.scale = glm::vec3(ele["SCALE"][0], ele["SCALE"][1], ele["SCALE"][2]);

        modelTrans.transform = utilityCore::buildTransformationMatrix(
            modelTrans.translation, modelTrans.rotation, modelTrans.scale);
        modelTrans.inverseTransform = glm::inverse(modelTrans.transform);
        modelTrans.invTranspose = glm::inverseTranspose(modelTrans.transform);

        new_obj.Transform = modelTrans;

        objects.emplace_back(new_obj);
    }
}

void Scene::loadSkybox()
{
    std::string line;
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        std::cout << "Loading Skybox " << line << " ..." << std::endl;
        environmentMapPath = line;
    }
}

namespace std {
    template <>
    struct hash<std::pair<glm::vec3, glm::vec2>> {
        std::size_t operator()(const std::pair<glm::vec3, glm::vec2>& vertex) const {
            return ((hash<float>()(vertex.first.x) ^
                (hash<float>()(vertex.first.y) << 1)) >> 1) ^
                (hash<float>()(vertex.first.z) << 1) ^ (hash<float>()(vertex.second.x) << 2) ^ ((hash<float>()(vertex.second.y) << 2) >> 2);
        }
    };

}

void Scene::LoadAllTexturesToGPU()
{
    if (environmentMapPath != "")
    {
        cudaTextureObject_t* texObj = &skyboxTextureObj;
        assert(!strToTextureObj.count(environmentMapPath));

        loadTextureFromFile(environmentMapPath, texObj, &environmentMapData);
        strToTextureObj[environmentMapPath] = *texObj;
    }


    for (auto& p : LoadTextureFromFileJobs)
    {
        cudaTextureObject_t tmp_tex;
        loadTextureFromFile(p.path, &tmp_tex);
        LoadMaterialJobs[p.matId].params.insert_texture(p.matTextureKey, tmp_tex);
    }
    /*
    for (auto& p : LoadTextureFromMemoryJobs)
    {
        Material& mat = materials[p.matIndex];
        cudaTextureObject_t* texObj = nullptr;
        switch (p.texType)
        {
        case TextureType::color:
            texObj = &mat.baseColorMap;
            break;
        case TextureType::normal:
            texObj = &mat.normalMap;
            break;
        case TextureType::metallicroughness:
            texObj = &mat.metallicRoughnessMap;
            break;
        }
        LoadTextureFromMemory(p.buffer, p.width, p.height, p.bits, p.component, texObj);
        delete[] p.buffer;
    }*/
}

void Scene::LoadAllMaterialsToGPU(Allocator alloc)
{
    for (auto& job : LoadMaterialJobs)
    {
        if (job.type == "dielectric" || job.type == "conductor")
        {
            if (job.params.get_string("eta") != std::string())
            {
                SpectrumPtr eta = spec::get_named_spectrum(job.params.get_string("eta"));
                job.params.insert_spectrum("eta", eta);
            }
            else if (job.params.get_vec3("eta") != glm::vec3(0.0f))
            {
                glm::vec3 eta = job.params.get_vec3("eta");
                job.params.insert_spectrum("eta", alloc.new_object<RGBUnboundedSpectrum>(*RGBColorSpace::sRGB, eta));
            }
            else if (job.params.get_float("eta") != 0.0f)
            {
                float eta = job.params.get_float("eta");
                job.params.insert_spectrum("eta", alloc.new_object<ConstantSpectrum>(eta));
            }

            if (job.params.get_string("k") != std::string())
            {
                SpectrumPtr k = spec::get_named_spectrum(job.params.get_string("k"));
                job.params.insert_spectrum("k", k);
            }
            else if (job.params.get_vec3("k") != glm::vec3(0.0f))
            {
                glm::vec3 k = job.params.get_vec3("k");
                job.params.insert_spectrum("k", alloc.new_object<RGBUnboundedSpectrum>(*RGBColorSpace::sRGB, k));
            }
            else if (job.params.get_float("k") != 0.0f)
            {
                float k = job.params.get_float("k");
                job.params.insert_spectrum("k", alloc.new_object<ConstantSpectrum>(k));
            }
        }
        job.params.insert_ptr("colorSpace", RGBColorSpace::sRGB);
        materials.emplace_back(MaterialPtr::create(job.type, job.params, alloc));
    }
}

void Scene::LoadAllMediaToGPU(Allocator alloc)
{
    for (auto& job : LoadMediumJobs)
    {
        if (job.type == "nanovdb" || job.type == "homogeneous")
        {
            glm::vec3 sigma_a_rgb = job.params.get_vec3("sigma_a_rgb");
            SpectrumPtr sigma_a = alloc.new_object<RGBUnboundedSpectrum>(*RGBColorSpace::ACES2065_1, sigma_a_rgb);
            auto sigma_a__ = sigma_a.Cast<RGBUnboundedSpectrum>();
            job.params.insert_spectrum("sigma_a", sigma_a__);

            glm::vec3 sigma_s_rgb = job.params.get_vec3("sigma_s_rgb");
            SpectrumPtr sigma_s = alloc.new_object<RGBUnboundedSpectrum>(*RGBColorSpace::ACES2065_1, sigma_s_rgb);
            auto sigma_s__ = sigma_s.Cast<RGBUnboundedSpectrum>();
            job.params.insert_spectrum("sigma_s", sigma_s__);
        }
        else
        {
            throw std::runtime_error("Not implemented!");
        }
        media.emplace_back(MediumPtr::create(job.type, job.params, job.world_from_medium, alloc));
    }
}

void Scene::LoadAllLightsToGPU(Allocator alloc)
{
    if (environmentMapData.data.size() > 0)
    {
        assert(environmentMapData.channelBits = 32);
        float* dataPtr = reinterpret_cast<float*>(environmentMapData.data.data());
        std::vector<float> luminanceData;
        int totalSize = environmentMapData.width * environmentMapData.height;
        luminanceData.reserve(totalSize);
        for (int i = 0; i < environmentMapData.width * environmentMapData.height * environmentMapData.channels; i += environmentMapData.channels)
        {
            glm::vec3 rgb(dataPtr[i], dataPtr[i + 1], dataPtr[i + 2]);
            rgb = min(rgb * environmentMapLuminScale, environmentMapMaxLumin);
            luminanceData.emplace_back(math::simple_rgb_to_lumin(rgb.r, rgb.g, rgb.b));
        }
        assert(luminanceData.size() == totalSize);
        BundledParams params;
        params.insert_float("scale", environmentMapLuminScale);
        params.insert_ptr("illumFunc", luminanceData.data());
        params.insert_int("width", environmentMapData.width);
        params.insert_int("height", environmentMapData.height);
        params.insert_texture("textureObject", strToTextureObj[environmentMapPath]);
        params.insert_vec3("maxRadiance", environmentMapMaxLumin);
        skyboxLight = ImageInfiniteLight::create(params, alloc);
        lights.emplace_back(skyboxLight);
    }
    for (int i=0;i<primitives.size();i++)
    {
        auto& prim = primitives[i];
        int matID = objects[prim.objID].materialid;
        if (matID != -1 && materials[matID].Is<EmissiveMaterial>())
        {
            BundledParams params;
            params.insert_int("primitiveID", i);
            EmissiveMaterial* mat = materials[matID].Cast<EmissiveMaterial>();
            // TODO: possible mem leak
            SpectrumPtr spec = alloc.new_object<RGBIlluminantSpectrum>(*mat->get_colorspace(), mat->get_rgb());
            params.insert_spectrum("Le_spec", spec);
            prim.lightID = static_cast<uint32_t>(lights.size());
            lights.emplace_back(DiffuseAreaLight::create(params, alloc));
        }
    }
}

void Scene::LoadAllMeshesToGPU(Allocator alloc)
{
    m_dev_triangleMeshes = alloc.allocate<TriangleMesh>(m_triangleMeshes.size());
    for (int i=0;i<m_triangleMeshes.size();i++)
    {
        const auto& mesh = m_triangleMeshes[i];
        TriangleMesh& dev_mesh = m_dev_triangleMeshes[i];
        dev_mesh.m_vertices = alloc.allocate<glm::vec3>(mesh.m_vertices.size());
        memcpy(dev_mesh.m_vertices, mesh.m_vertices.data(), sizeof(glm::vec3) * mesh.m_vertices.size());
        dev_mesh.m_triangles = alloc.allocate<glm::ivec3>(mesh.m_triangles.size());
        memcpy(dev_mesh.m_triangles, mesh.m_triangles.data(), sizeof(glm::ivec3) * mesh.m_triangles.size());
        if (mesh.m_normals.size())
        {
            dev_mesh.m_normals = alloc.allocate<glm::vec3>(mesh.m_normals.size());
            memcpy(dev_mesh.m_normals, mesh.m_normals.data(), sizeof(glm::vec3) * mesh.m_normals.size());
        }
        if (mesh.m_uvs.size())
        {
            dev_mesh.m_uvs = alloc.allocate<glm::vec2>(mesh.m_uvs.size());
            memcpy(dev_mesh.m_uvs, mesh.m_uvs.data(), sizeof(glm::vec2) * mesh.m_uvs.size());
        }
    }
}


Scene::~Scene()
{
    /*for (auto& p : strToTextureObj)
    {
        cudaDestroyTextureObject(materials[p.second].baseColorMap);
    }
    for (auto& p : textureDataPtrs)
    {
        cudaFreeArray(p);
    }*/
}

void Scene::LoadTextureFromMemory(void* data, int width, int height, int bits, int channels, cudaTextureObject_t* texObj)
{
    assert(channels == 4);
    cudaError_t err;
    size_t dataSize = width * height * 4 * (bits >> 3);
    cudaArray_t cuArray;
    cudaChannelFormatKind format = bits == 8 ? cudaChannelFormatKindUnsigned : cudaChannelFormatKindFloat;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(bits, bits, bits, bits, format);
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    textureDataPtrs.emplace_back(cuArray);
    cudaMemcpyToArray(cuArray, 0, 0, data, width * height * 4 * (bits >> 3), cudaMemcpyHostToDevice);

    
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    resDesc.res.linear.desc = cudaCreateChannelDesc(bits, bits, bits, bits, format);
    resDesc.res.linear.sizeInBytes = dataSize;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = bits == 8 ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
    texDesc.sRGB = 1;
    cudaCreateTextureObject(texObj, &resDesc, &texDesc, NULL);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void Scene::loadTextureFromFile(const std::string& texturePath, cudaTextureObject_t* texObj, RawTextureData* ret_data)
{
    int width, height, channels;
    std::string ext = texturePath.substr(texturePath.find_last_of('.') + 1);

    if (ext == "hdr") {
        float* data = stbi_loadf(texturePath.c_str(), &width, &height, &channels, 4);
        if (ret_data) {
            size_t dataSize = width * height * 4 * sizeof(float);
            ret_data->width = width;
            ret_data->height = height;
            ret_data->channels = 4;
            ret_data->channelBits = 32;
            ret_data->data.resize(dataSize);
            memcpy(ret_data->data.data(), data, dataSize);
        }
        if (data) {
            LoadTextureFromMemory(data, width, height, 32, 4, texObj);
            stbi_image_free(data);
        }
        else {
            printf("Failed to load HDR image: %s\n", stbi_failure_reason());
        }
    }
    else if (ext == "exr") {
        float* out_rgba = nullptr;
        const char* err = nullptr;
        int ret = LoadEXR(&out_rgba, &width, &height, texturePath.c_str(), &err);
        if (ret == TINYEXR_SUCCESS) {
            channels = 4; // TinyEXR outputs RGBA
            if (ret_data) {
                size_t dataSize = width * height * 4 * sizeof(float);
                ret_data->width = width;
                ret_data->height = height;
                ret_data->channels = 4;
                ret_data->channelBits = 32;
                ret_data->data.resize(dataSize);
                memcpy(ret_data->data.data(), out_rgba, dataSize);
            }
            LoadTextureFromMemory(out_rgba, width, height, 32, 4, texObj);
            free(out_rgba);
        }
        else {
            fprintf(stderr, "Failed to load EXR image: %s\n", err);
            FreeEXRErrorMessage(err);
        }
    }
    else {
        unsigned char* data = stbi_load(texturePath.c_str(), &width, &height, &channels, 4);
        if (ret_data) {
            size_t dataSize = width * height * 4 * sizeof(unsigned char);
            ret_data->width = width;
            ret_data->height = height;
            ret_data->channels = 4;
            ret_data->channelBits = 8;
            ret_data->data.resize(dataSize);
            memcpy(ret_data->data.data(), data, dataSize);
        }
        if (data) {
            LoadTextureFromMemory(data, width, height, 8, 4, texObj);
            stbi_image_free(data);
        }
        else {
            printf("Failed to load image: %s\n", stbi_failure_reason());
        }
    }
}

static void GLTFNodeGetLocalTransform(tinygltf::Node& node, glm::mat4& localTransform)
{
    if (node.matrix.size() == 16)
    {
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
            {
                localTransform[j][k] = node.matrix[k + j * 4];
            }
    }
    else
    {
        auto& rot = node.rotation;
        auto& trans = node.translation;
        auto& scale = node.scale;
        glm::mat4 transM(1.0f);
        glm::mat4 rotM(1.0f);
        glm::mat4 scaleM(1.0f);
        if (rot.size())
            rotM = glm::mat4(glm::quat(rot[3], rot[0], rot[1], rot[2]));
        if (trans.size())
            transM = glm::translate(transM, glm::vec3(trans[0], trans[1], trans[2]));
        if (scale.size())
            scaleM = glm::scale(scaleM, glm::vec3(scale[0], scale[1], scale[2]));
        localTransform = transM * rotM * scaleM;
    }
}

static void GLTFNodetopologicalSort(std::vector<tinygltf::Node>& nodes, std::vector<int>& sortedIdx)
{
    std::vector<int> inDegs(nodes.size());
    std::queue<int> q;
    for (auto& node : nodes)
    {
        for (auto& chld : node.children)
        {
            inDegs[chld]++;
        }
    }
    for (int i = 0; i < inDegs.size(); i++)
    {
        if (inDegs[i] == 0)
        {
            q.emplace(i);
        }
    }
    while (!q.empty())
    {
        auto p = q.front(); q.pop();
        sortedIdx.emplace_back(p);
        for (auto& chld : nodes[p].children)
        {
            inDegs[chld]--;
            if (inDegs[chld] == 0)
                q.emplace(chld);
        }
    }
    
}

static void GLTFNodeGetGlobalTransform(std::vector<tinygltf::Node>& nodes, int curr, std::unordered_map<int, glm::mat4>& rec, glm::mat4 parentTrans = glm::mat4(1.0))
{
    auto& node = nodes[curr];
    glm::mat4 localTrans;
    GLTFNodeGetLocalTransform(node, localTrans);
    if (!rec.count(curr))
    {
        rec[curr] = parentTrans * localTrans;
    }
    else return;
    for (int& chld : node.children)
    {
        GLTFNodeGetGlobalTransform(nodes, chld, rec, rec[curr]);
    }
}

//int MikkTSpaceGetNumFaces(const SMikkTSpaceContext* pContext)
//{
//    MikkTSpaceHelper* helperStruct = (MikkTSpaceHelper*)pContext->m_pUserData;
//    auto& obj = helperStruct->scene->objects[helperStruct->i];
//    return obj.triangleEnd - obj.triangleStart;
//}
//
//int MikkTSpaceGetNumVerticesOfFace(const SMikkTSpaceContext* pContext, const int iFace) {
//    // return the number of vertices for the i'th face.
//    return 3;
//}
//
//void MikkTSpaceGetPosition(const SMikkTSpaceContext* pContext, float fvPosOut[], const int iFace, const int iVert) {
//    // fill fvPosOut with the position of vertex iVert of face iFace
//    MikkTSpaceHelper* helperStruct = (MikkTSpaceHelper*)pContext->m_pUserData;
//    auto scene = helperStruct->scene;
//    auto& obj = scene->objects[helperStruct->i];
//    int triIdx = obj.triangleStart + iFace;
//    auto& tri = scene->triangles[triIdx];
//    auto& pos = helperStruct->scene->verticies[tri[iVert]];
//    fvPosOut[0] = pos[0];
//    fvPosOut[1] = pos[1];
//    fvPosOut[2] = pos[2];
//}
//
//void MikkTSpaceGetNormal(const SMikkTSpaceContext* pContext, float fvNormOut[], const int iFace, const int iVert) {
//    // fill fvNormOut with the normal of vertex iVert of face iFace
//    MikkTSpaceHelper* helperStruct = (MikkTSpaceHelper*)pContext->m_pUserData;
//    auto scene = helperStruct->scene;
//    auto& obj = scene->objects[helperStruct->i];
//    int triIdx = obj.triangleStart + iFace;
//    auto& tri = scene->triangles[triIdx];
//    auto& norm = scene->m_normals[tri[iVert]];
//    fvNormOut[0] = norm[0];
//    fvNormOut[1] = norm[1];
//    fvNormOut[2] = norm[2];
//}
//
//void MikkTSpaceGetTexCoord(const SMikkTSpaceContext* pContext, float fvTexcOut[], const int iFace, const int iVert) {
//    // fill fvTexcOut with the texture coordinate of vertex iVert of face iFace
//    MikkTSpaceHelper* helperStruct = (MikkTSpaceHelper*)pContext->m_pUserData;
//    auto scene = helperStruct->scene;
//    auto& obj = scene->objects[helperStruct->i];
//    int triIdx = obj.triangleStart + iFace;
//    auto& tri = scene->triangles[triIdx];
//    auto& uv = scene->uvs[tri[iVert]];
//    fvTexcOut[0] = uv[0];
//    fvTexcOut[1] = uv[1];
//}
//
//void MikkTSpaceSetTSpaceBasic(const SMikkTSpaceContext* pContext, const float fvTangent[], const float fSign, const int iFace, const int iVert) {
//    // store the tangent and sign to your mesh vertex
//    MikkTSpaceHelper* helperStruct = (MikkTSpaceHelper*)pContext->m_pUserData;
//    auto scene = helperStruct->scene;
//    auto& obj = scene->objects[helperStruct->i];
//    int triIdx = obj.triangleStart + iFace;
//    auto& tri = scene->triangles[triIdx];
//    auto& tangent = scene->tangents[tri[iVert]];
//    tangent[0] = fvTangent[0];
//    tangent[1] = fvTangent[1];
//    tangent[2] = fvTangent[2];
//    scene->fSigns[tri[iVert]] = fSign;
//}


// load model using tinyobjloader and tinygltf
bool Scene::loadModel(const std::string& modelPath, int objectid, bool useVertexNormal)
{
    using namespace std;
    cout << "Loading Model " << modelPath << " ..." << endl;
    string postfix = modelPath.substr(modelPath.find_last_of('.') + 1);
    if (postfix == "obj")//load obj
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> aShapes;
        std::vector<tinyobj::material_t> aMaterials;
        std::string warn;
        std::string err;
        std::string mtlPath = modelPath.substr(0, modelPath.find_last_of('/') + 1);
        bool ret = tinyobj::LoadObj(&attrib, &aShapes, &aMaterials, &warn, &err, modelPath.c_str(), mtlPath.c_str());
        if (!warn.empty()) std::cout << warn << std::endl;

        if (!err.empty()) std::cerr << err << std::endl;

        if (!ret)  return false;

        int matOffset = materials.size();
        
        for (const auto& mat : aMaterials)
        {
            std::string type = "diffuse";
            BundledParams params;
            params.insert_vec3("albedo", glm::vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]));
            // TODO
            if (!mat.diffuse_texname.empty())
            {
                LoadTextureFromFileJobs.emplace_back(mtlPath + mat.diffuse_texname, static_cast<int>(materials.size()));
            }
            std::string line;
            utilityCore::safeGetline(fp_in, line);

        }
        int modelStartIdx = objects.size();
        for (const auto& shape : aShapes)
        {
            TriangleMeshData newMesh;
            Object model;
            // TODO: check if this is necessary
            std::unordered_map<std::pair<glm::vec3, glm::vec2>, uint32_t> vertexSet;
            for (const auto& index : shape.mesh.indices)
            {
                glm::vec3 tmp_pos;
                tmp_pos.x = attrib.vertices[3 * index.vertex_index + 0];
                tmp_pos.y = attrib.vertices[3 * index.vertex_index + 1];
                tmp_pos.z = attrib.vertices[3 * index.vertex_index + 2];
                glm::vec2 tmp_uv;
                tmp_uv.x = index.texcoord_index >= 0 ? attrib.texcoords[2 * index.texcoord_index + 0] : -1.0;
                tmp_uv.y = index.texcoord_index >= 0 ? attrib.texcoords[2 * index.texcoord_index + 1] : -1.0;
                glm::vec3 tmp_normal;
                if (useVertexNormal)
                {
                    tmp_normal.x = index.normal_index >= 0 ? attrib.normals[3 * index.normal_index + 0] : -1.0;
                    tmp_normal.y = index.normal_index >= 0 ? attrib.normals[3 * index.normal_index + 1] : -1.0;
                    tmp_normal.z = index.normal_index >= 0 ? attrib.normals[3 * index.normal_index + 2] : -1.0;
                }
                auto vertexInfo = std::make_pair(tmp_pos, tmp_uv);
                if (!vertexSet.count(vertexInfo))
                {
                    vertexSet[vertexInfo] = (uint32_t)vertexSet.size();
                    newMesh.m_vertices.emplace_back(tmp_pos);
                    newMesh.m_uvs.emplace_back(tmp_uv);
                    if (useVertexNormal) newMesh.m_normals.emplace_back(tmp_normal);
                }
            }
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
            {
                int fv = shape.mesh.num_face_vertices[f];
                assert(fv == 3);
                glm::ivec3 triangle;
                for (size_t v = 0; v < fv; v++) {
                    tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                    glm::vec3 tmp_pos;
                    tmp_pos.x = attrib.vertices[3 * idx.vertex_index + 0];
                    tmp_pos.y = attrib.vertices[3 * idx.vertex_index + 1];
                    tmp_pos.z = attrib.vertices[3 * idx.vertex_index + 2];
                    glm::vec2 tmp_uv;
                    tmp_uv.x = idx.texcoord_index >= 0 ? attrib.texcoords[2 * idx.texcoord_index + 0] : -1.0;
                    tmp_uv.y = idx.texcoord_index >= 0 ? attrib.texcoords[2 * idx.texcoord_index + 1] : -1.0;
                    auto vertexInfo = std::make_pair(tmp_pos, tmp_uv);
                    triangle[v] = vertexSet[vertexInfo];
                }
                newMesh.m_triangles.emplace_back(triangle);
                index_offset += fv;
            }
            model.type = TRIANGLE_MESH;
            model.meshId = static_cast<int>(m_triangleMeshes.size());
            model.materialid = shape.mesh.material_ids[0] + matOffset;//Assume per mesh material
            objects.emplace_back(model);
            m_triangleMeshes.emplace_back(std::move(newMesh));
        }

        int modelEndIdx = objects.size();

        std::string line;
        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            int mID = atoi(tokens[1].c_str());
            if (mID != -1)
            {
                for (int i = modelStartIdx; i != modelEndIdx; i++)
                {
                    objects[i].materialid = mID;
                }
                cout << "Connecting Object " << objectid << " to Material " << mID << "..." << endl;
            }
        }

        ObjectTransform modelTrans;
        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) 
        {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) 
            {
                modelTrans.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) 
            {
                modelTrans.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "SCALE") == 0) 
            {
                modelTrans.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        modelTrans.transform = utilityCore::buildTransformationMatrix(
            modelTrans.translation, modelTrans.rotation, modelTrans.scale);
        modelTrans.inverseTransform = glm::inverse(modelTrans.transform);
        modelTrans.invTranspose = glm::inverseTranspose(modelTrans.transform);

        for (int i = modelStartIdx; i != modelEndIdx; i++)
        {
            objects[i].Transform = modelTrans;
        }

    }
    //else//gltf
    //{
    //    tinygltf::Model model;
    //    tinygltf::TinyGLTF loader;
    //    std::string err;
    //    std::string warn;

    //    bool ret;
    //    if (postfix == "glb")
    //        ret = loader.LoadBinaryFromFile(&model, &err, &warn, modelPath.c_str());
    //    else if (postfix == "gltf")
    //        ret = loader.LoadASCIIFromFile(&model, &err, &warn, modelPath.c_str());
    //    else assert(0);//unexpected format

    //    if (!warn.empty())  std::cout << "Tiny GLTF Warn: " << warn << std::endl;

    //    if (!err.empty()) std::cout << "Tiny GLTF Err: " << err << std::endl;

    //    if (!ret) 
    //    {
    //        std::cout << "Failed to parse glTF" << std::endl;
    //        return -1;
    //    }
    //    int matOffset = materials.size();
    //    //load materials
    //    for (size_t i = 0; i < model.materials.size(); i++)
    //    {
    //        Material newMat;
    //        newMat.type = metallicWorkflow;
    //        auto& gltfMat = model.materials[i];
    //        auto& pbr = gltfMat.pbrMetallicRoughness;
    //        newMat.color[0] = pbr.baseColorFactor[0];
    //        newMat.color[1] = pbr.baseColorFactor[1];
    //        newMat.color[2] = pbr.baseColorFactor[2];
    //        newMat.roughness = pbr.roughnessFactor;
    //        newMat.metallic = pbr.metallicFactor;
    //        auto& baseColorTex = pbr.baseColorTexture;
    //        auto& metallicRoughnessTex = pbr.metallicRoughnessTexture;
    //        auto& normalTex = gltfMat.normalTexture;
    //        if (baseColorTex.index != -1)
    //        {
    //            assert(baseColorTex.texCoord == 0);//multi texcoord is not supported
    //            auto& tex = model.textures[baseColorTex.index];
    //            auto& image = model.images[tex.source];
    //            char* tmpBuffer = new char[image.image.size()];
    //            memcpy(tmpBuffer, &image.image[0], image.image.size());
    //            LoadTextureFromMemoryJobs.emplace_back(tmpBuffer, materials.size(), TextureType::color, image.width, image.height, image.bits, image.component);
    //        }
    //        if (metallicRoughnessTex.index != -1)
    //        {
    //            assert(metallicRoughnessTex.texCoord == 0);//multi texcoord is not supported
    //            auto& tex = model.textures[metallicRoughnessTex.index];
    //            auto& image = model.images[tex.source];
    //            char* tmpBuffer = new char[image.image.size()];
    //            memcpy(tmpBuffer, &image.image[0], image.image.size());
    //            LoadTextureFromMemoryJobs.emplace_back(tmpBuffer, materials.size(), TextureType::metallicroughness, image.width, image.height, image.bits, image.component);
    //        }
    //        if (normalTex.index != -1)
    //        {
    //            assert(normalTex.texCoord == 0);//multi texcoord is not supported
    //            auto& tex = model.textures[normalTex.index];
    //            auto& image = model.images[tex.source];
    //            char* tmpBuffer = new char[image.image.size()];
    //            memcpy(tmpBuffer, &image.image[0], image.image.size());
    //            LoadTextureFromMemoryJobs.emplace_back(tmpBuffer, materials.size(), TextureType::normal, image.width, image.height, image.bits, image.component);
    //        }
    //        if (gltfMat.extensions.count("KHR_materials_transmission")|| gltfMat.extensions.count("KHR_materials_volume")||gltfMat.alphaMode=="BLEND")//limited support for translucency
    //        {
    //            newMat.type = frenselSpecular;
    //            //newMat.color = gltfMat.extensions["KHR_materials_volume"].Get("attenuationColor").GetNumberAsDouble();
    //            newMat.color = glm::vec3(0.98f);
    //            if (gltfMat.extensions.count("KHR_materials_ior"))
    //                newMat.indexOfRefraction = gltfMat.extensions["KHR_materials_ior"].Get("ior").GetNumberAsDouble();
    //            else
    //                newMat.indexOfRefraction = 1.5f;
    //        }
    //        materials.emplace_back(newMat);
    //    }

    //   
    //    std::unordered_map<int, glm::mat4> globalTransRec;
    //    std::vector<int> sortedIdx;
    //    GLTFNodetopologicalSort(model.nodes, sortedIdx);

    //    for (size_t i = 0; i < model.nodes.size(); ++i)
    //    {
    //        int curr = sortedIdx[i];
    //        GLTFNodeGetGlobalTransform(model.nodes, curr, globalTransRec);
    //    }
    //    
    //    int modelStartIdx = objects.size();

    //    for (size_t i = 0; i < model.nodes.size(); ++i) 
    //    {
    //        tinygltf::Node& node = model.nodes[i];
    //        if (node.camera != -1 || node.mesh == -1) continue;//ignore GLTF's camera
    //        auto& mesh = model.meshes[node.mesh];
    //        const glm::mat4& trans = globalTransRec[i];
    //        
    //        for (auto& primtive : mesh.primitives)
    //        {
    //            int triangleIdxOffset = verticies.size();//each gltf primitive assume a index starts with 0
    //            assert(primtive.mode == TINYGLTF_MODE_TRIANGLES);
    //            Object newModel;
    //            newModel.type = TRIANGLE_MESH;
    //            newModel.triangleStart = triangles.size();
    //            newModel.materialid = primtive.material + matOffset;
    //            newModel.Transform.transform = trans;
    //            
    //            int indicesAccessorIdx = primtive.indices;
    //            int positionAccessorIdx = -1, normalAccessorIdx = -1, texcoordAccessorIdx = -1;
    //            if (primtive.attributes.count("POSITION"))
    //            {
    //                positionAccessorIdx = primtive.attributes["POSITION"];
    //            }
    //            if (primtive.attributes.count("NORMAL")) 
    //            {
    //                normalAccessorIdx = primtive.attributes["NORMAL"];
    //            }
    //            if (primtive.attributes.count("TEXCOORD_0"))
    //            {
    //                texcoordAccessorIdx = primtive.attributes["TEXCOORD_0"];
    //            }
    //            assert(positionAccessorIdx != -1 && indicesAccessorIdx != -1);
    //            //Load indices
    //            auto& indicesAccessor = model.accessors[indicesAccessorIdx];
    //            if (indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_SHORT || indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
    //            {
    //                auto& bView = model.bufferViews[indicesAccessor.bufferView];
    //                size_t stride = bView.byteStride ? bView.byteStride : (indicesAccessor.type & 0xF) * sizeof(short);
    //                unsigned char* ptr = &model.buffers[bView.buffer].data[0] + bView.byteOffset + indicesAccessor.byteOffset;
    //                for (int i = 0; i < indicesAccessor.count; i += 3)
    //                {
    //                    glm::ivec3 tri;
    //                    tri.x = *(unsigned short*)(ptr + (i + 0) * stride) + triangleIdxOffset;
    //                    tri.y = *(unsigned short*)(ptr + (i + 1) * stride) + triangleIdxOffset;
    //                    tri.z = *(unsigned short*)(ptr + (i + 2) * stride) + triangleIdxOffset;
    //                    triangles.emplace_back(tri);
    //                }
    //            }
    //            else if (indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_INT || indicesAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
    //            {
    //                auto& bView = model.bufferViews[indicesAccessor.bufferView];
    //                size_t stride = bView.byteStride ? bView.byteStride : (indicesAccessor.type & 0xF) * sizeof(int);
    //                unsigned char* ptr = &model.buffers[bView.buffer].data[0] + bView.byteOffset + indicesAccessor.byteOffset;
    //                for (int i = 0; i < indicesAccessor.count; i += 3)
    //                {
    //                    glm::ivec3 tri;
    //                    tri.x = *(unsigned int*)(ptr + (i + 0) * stride) + triangleIdxOffset;
    //                    tri.y = *(unsigned int*)(ptr + (i + 1) * stride) + triangleIdxOffset;
    //                    tri.z = *(unsigned int*)(ptr + (i + 2) * stride) + triangleIdxOffset;
    //                    triangles.emplace_back(tri);
    //                }
    //            }
    //            else assert(0);//unexpected
    //            newModel.triangleEnd = triangles.size();
    //            objects.emplace_back(newModel);
    //            //Load position
    //            auto& positionAccessor = model.accessors[positionAccessorIdx];
    //            if (positionAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
    //            {
    //                assert(positionAccessor.type = TINYGLTF_TYPE_VEC3);
    //                auto& bView = model.bufferViews[positionAccessor.bufferView];
    //                size_t stride = bView.byteStride ? bView.byteStride : (positionAccessor.type & 0xF) * sizeof(float);
    //                unsigned char* ptr = &model.buffers[bView.buffer].data[0] + bView.byteOffset + positionAccessor.byteOffset;
    //                for (int i = 0; i < positionAccessor.count; i++)
    //                {
    //                    glm::vec3 pos;
    //                    pos.x = *(float*)(ptr + (i * stride + sizeof(float) * 0));
    //                    pos.y = *(float*)(ptr + (i * stride + sizeof(float) * 1));
    //                    pos.z = *(float*)(ptr + (i * stride + sizeof(float) * 2));
    //                    verticies.emplace_back(pos);
    //                }
    //            }
    //            else assert(0);//unexpected
    //            //Load normals
    //            auto& normalAccessor = model.accessors[normalAccessorIdx];
    //            if (useVertexNormal && normalAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
    //            {
    //                assert(normalAccessor.type = TINYGLTF_TYPE_VEC3);
    //                auto& bView = model.bufferViews[normalAccessor.bufferView];
    //                size_t stride = bView.byteStride ? bView.byteStride : (normalAccessor.type & 0xF) * sizeof(float);
    //                unsigned char* ptr = &model.buffers[bView.buffer].data[0] + bView.byteOffset + normalAccessor.byteOffset;
    //                for (int i = 0; i < normalAccessor.count; i++)
    //                {
    //                    glm::vec3 normal;
    //                    normal.x = *(float*)(ptr + (i * stride + sizeof(float) * 0));
    //                    normal.y = *(float*)(ptr + (i * stride + sizeof(float) * 1));
    //                    normal.z = *(float*)(ptr + (i * stride + sizeof(float) * 2));
    //                    normals.emplace_back(normal);
    //                }
    //            }
    //            else assert(0);//unexpected
    //            if(useVertexNormal)
    //                assert(verticies.size() == normals.size());
    //            //Load uv
    //            auto& texcoordAccessor = model.accessors[texcoordAccessorIdx];
    //            if (texcoordAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT)
    //            {
    //                assert(texcoordAccessor.type = TINYGLTF_TYPE_VEC2);
    //                auto& bView = model.bufferViews[texcoordAccessor.bufferView];
    //                size_t stride = bView.byteStride ? bView.byteStride : (texcoordAccessor.type & 0x7) * sizeof(float);
    //                unsigned char* ptr = &model.buffers[bView.buffer].data[0] + bView.byteOffset + texcoordAccessor.byteOffset;
    //                for (int i = 0; i < texcoordAccessor.count; i++)
    //                {
    //                    glm::vec2 uv;
    //                    uv.x = *(float*)(ptr + (i * stride + sizeof(float) * 0));
    //                    uv.y = *(float*)(ptr + (i * stride + sizeof(float) * 1));
    //                    uvs.emplace_back(uv);
    //                }
    //            }
    //            else assert(0);//unexpected
    //            assert(verticies.size() == uvs.size());
    //            if (useVertexNormal)
    //            {
    //                //use MikkTSpace to calculate tangents
    //                fSigns.resize(normals.size());
    //                tangents.resize(normals.size());
    //                SMikkTSpaceInterface interface = {
    //                    MikkTSpaceGetNumFaces,
    //                    MikkTSpaceGetNumVerticesOfFace,
    //                    MikkTSpaceGetPosition,
    //                    MikkTSpaceGetNormal,
    //                    MikkTSpaceGetTexCoord,
    //                    MikkTSpaceSetTSpaceBasic,
    //                    NULL,  // setTSpace. Can be NULL.
    //                };
    //                MikkTSpaceHelper helperStruct;
    //                helperStruct.i = objects.size() - 1;
    //                helperStruct.scene = this;
    //                SMikkTSpaceContext context = {
    //                    &interface,
    //                    &helperStruct,  
    //                };
    //                genTangSpaceDefault(&context);
    //            }
    //        }
    //    }
    //    int modelEndIdx = objects.size();

    //    std::string line;
    //    //link material
    //    utilityCore::safeGetline(fp_in, line);
    //    if (!line.empty() && fp_in.good()) {
    //        vector<string> tokens = utilityCore::tokenizeString(line);
    //        int mID = atoi(tokens[1].c_str());
    //        if (mID != -1)
    //        {
    //            for (int i = modelStartIdx; i != modelEndIdx; i++)
    //            {
    //                objects[i].materialid = mID;
    //            }
    //            cout << "Connecting Geom " << objectid << " to Material " << mID << "..." << endl;
    //        }
    //    }

    //    ObjectTransform modelTrans;
    //    //load transformations
    //    utilityCore::safeGetline(fp_in, line);
    //    while (!line.empty() && fp_in.good())
    //    {
    //        vector<string> tokens = utilityCore::tokenizeString(line);

    //        //load tranformations
    //        if (strcmp(tokens[0].c_str(), "TRANS") == 0)
    //        {
    //            modelTrans.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
    //        }
    //        else if (strcmp(tokens[0].c_str(), "ROTAT") == 0)
    //        {
    //            modelTrans.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
    //        }
    //        else if (strcmp(tokens[0].c_str(), "SCALE") == 0)
    //        {
    //            modelTrans.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
    //        }

    //        utilityCore::safeGetline(fp_in, line);
    //    }

    //    modelTrans.transform = utilityCore::buildTransformationMatrix(
    //        modelTrans.translation, modelTrans.rotation, modelTrans.scale);

    //    for (int i = modelStartIdx; i != modelEndIdx; i++)
    //    {
    //        objects[i].Transform.transform = modelTrans.transform * objects[i].Transform.transform;
    //        objects[i].Transform.inverseTransform = glm::inverse(objects[i].Transform.transform);
    //        objects[i].Transform.invTranspose = glm::inverseTranspose(objects[i].Transform.transform);
    //    }
    //}

    return true;
}

bool Scene::loadPly(Object* newObj, const std::string& path)
{
    // TODO: preload the data if possible
    // see https://github.com/ddiakopoulos/tinyply/blob/master/source/example.cpp

    std::ifstream file_stream(path, std::ios::binary);

    if (!file_stream.is_open() || !file_stream.good())
    {
        throw std::runtime_error("Failed to open" + path);
    }

    tinyply::PlyFile file;
    file.parse_header(file_stream);

    std::shared_ptr<tinyply::PlyData> vertices, normals, colors, texcoords, faces, tripstrip, tangents;

    try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
    catch (const std::exception& e) { 
        std::cerr << "tinyply exception: " << e.what() << std::endl; 
    }

    try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
    catch (const std::exception& e) { 
        std::cerr << "tinyply exception: " << e.what() << std::endl; 
    }

    try {
        tangents = file.request_properties_from_element("vertex", { "tx", "ty", "tz" });
    }
    catch (const std::exception& e) {
        std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try { colors = file.request_properties_from_element("vertex", { "red", "green", "blue", "alpha" }); }
    catch (const std::exception& e) { 
        std::cerr << "tinyply exception: " << e.what() << std::endl; 
    }

    try { texcoords = file.request_properties_from_element("vertex", { "u", "v" }); }
    catch (const std::exception& e) { 
        std::cerr << "tinyply exception: " << e.what() << std::endl; 
    }

    // Providing a list size hint (the last argument) is a 2x performance improvement. If you have 
    // arbitrary ply files, it is best to leave this 0. 
    try { faces = file.request_properties_from_element("face", { "vertex_indices" }, 0); }
    catch (const std::exception& e) { 
        std::cerr << "tinyply exception: " << e.what() << std::endl; 
    }

    // Tristrips must always be read with a 0 list size hint (unless you know exactly how many elements
    // are specifically in the file, which is unlikely); 
    try { tripstrip = file.request_properties_from_element("tristrips", { "vertex_indices" }, 0); }
    catch (const std::exception& e) {
        std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    file.read(file_stream);

    newObj->type = TRIANGLE_MESH;

    TriangleMeshData newMesh;

    const size_t numVerticesBytes = vertices->buffer.size_bytes();
    assert(vertices->t == tinyply::Type::FLOAT32);
    float* vert_buffer = (float*)vertices->buffer.get();
    for (size_t i = 0; i < vertices->count; i++)
    {
        glm::vec3 vert(vert_buffer[3 * i], vert_buffer[3 * i + 1], vert_buffer[3 * i + 2]);
        newMesh.m_vertices.emplace_back(vert);
    }
    
    assert(normals->t == tinyply::Type::FLOAT32);
    float* norm_buffer = (float*)normals->buffer.get();
    for (size_t i = 0; i < normals->count; i++)
    {
        glm::vec3 norm(norm_buffer[3 * i], norm_buffer[3 * i + 1], norm_buffer[3 * i + 2]);
        newMesh.m_normals.emplace_back(norm);
    }

    assert(texcoords->t == tinyply::Type::FLOAT32);
    float* uv_buffer = (float*)texcoords->buffer.get();
    for (size_t i = 0; i < texcoords->count; i++)
    {
        glm::vec2 uv(uv_buffer[2 * i], uv_buffer[2 * i + 1]);
        newMesh.m_uvs.emplace_back(uv);
    }

    assert(faces->t == tinyply::Type::INT32);
    uint32_t* idx_buffer = (uint32_t*)faces->buffer.get();
    for (size_t i = 0; i < faces->count; i++)
    {
        glm::ivec3 tri(idx_buffer[3 * i], idx_buffer[3 * i + 1], idx_buffer[3 * i + 2]);
        newMesh.m_triangles.emplace_back(tri);
    }
    newObj->meshId = static_cast<int>(m_triangleMeshes.size());
    m_triangleMeshes.emplace_back(std::move(newMesh));

    return true;
}

bool Scene::loadGeometry(const std::string& type, int objectid)
{
    using namespace std;
    string line;
    Object newGeom;
    //load geometry type
    if (type == "sphere") {
        std::cout << "Creating new sphere..." << endl;
        newGeom.type = SPHERE;
    }
    else if (type == "cube") {
        std::cout << "Creating new cube..." << endl;
        newGeom.type = CUBE;
    }

    //link material
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        newGeom.materialid = atoi(tokens[1].c_str());
        std::cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
    }

    //load transformations
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
            newGeom.Transform.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
            newGeom.Transform.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
            newGeom.Transform.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    newGeom.Transform.transform = utilityCore::buildTransformationMatrix(
        newGeom.Transform.translation, newGeom.Transform.rotation, newGeom.Transform.scale);
    newGeom.Transform.inverseTransform = glm::inverse(newGeom.Transform.transform);
    newGeom.Transform.invTranspose = glm::inverseTranspose(newGeom.Transform.transform);

    objects.push_back(newGeom);
    return true;
}

int Scene::loadObject(std::string objectid) {
    using namespace std;
    int id = atoi(objectid.c_str());
    std::cout << "Loading Object " << id << "..." << endl;
    string line;
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good())
    {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (tokens[0] == "geometry")//load geometry
        {
            loadGeometry(tokens[1], id);
        }
        else//load model
        {
            assert(tokens.size() == 3);
            assert(tokens[1] == "vnormal" || tokens[1] == "fnormal");
            bool use_vertex_normal = tokens[1] == "vnormal";
            loadModel(tokens[2], id, use_vertex_normal);
        }
    }
    return 1;
    
}



int Scene::loadCamera() {
    using namespace std;
    std::cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    std::cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(std::string materialid) {
    using namespace std;
    int id = atoi(materialid.c_str());
    if (id != LoadMaterialJobs.size()) {
        std::cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        std::cout << "Loading Material " << id << "..." << endl;
        //Material newMaterial;
        BundledParams params;
        Allocator alloc;
        std::string type;
        //load static properties
        while(true){
            string line;
            utilityCore::safeGetline(fp_in, line);
            if (line.size() == 0) break;
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "TYPE") == 0) {
                if (tokens[1] == "diffuse")
                    type = "diffuse";
                else if (tokens[1] == "frenselSpecular")
                    type = "dielectric";
                else if (tokens[1] == "conductor")
                    type = "conductor";
                else if (tokens[1] == "emitting")
                    type = "emissive";
                /*else if (tokens[1] == "metallicWorkflow")
                    newMaterial.type = metallicWorkflow;
                else if (tokens[1] == "blinnphong")
                    newMaterial.type = blinnphong;
                else if (tokens[1] == "asymMicrofacet")
                    newMaterial.type = asymMicrofacet;
                else
                    newMaterial.type = emitting;*/
            }
            else if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                params.insert_vec3("albedo", color);
            } 
            else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                float ior = atof(tokens[1].c_str());
                params.insert_float("eta", ior);
                //params.insert_spectrum("eta", alloc.new_object<ConstantSpectrum>(ior));
            }
            else if (strcmp(tokens[0].c_str(), "REFRIOR_NAMED") == 0) {
                params.insert_string("eta", tokens[1]);
            }
            else if (strcmp(tokens[0].c_str(), "REFRIOR_RGB") == 0) {
                glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                params.insert_vec3("eta", color);
                //params.insert_spectrum("eta", alloc.new_object<RGBUnboundedSpectrum>(*RGBColorSpace::sRGB, color));
            }
            else if (strcmp(tokens[0].c_str(), "REFRIOR_REAL_NAMED") == 0)
            {
                params.insert_string("eta", tokens[1]);
            }
            else if (strcmp(tokens[0].c_str(), "REFRIOR_IMAG_NAMED") == 0)
            {
                params.insert_string("k", tokens[1]);
            }
            else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                float emittance = atof(tokens[1].c_str());
                params.insert_float("emittance", emittance);
            }
            else if (strcmp(tokens[0].c_str(), "ROUGHNESS") == 0) {
                float roughness = atof(tokens[1].c_str());
                params.insert_float("roughness", roughness);
            }
            /*else if (strcmp(tokens[0].c_str(), "METALLIC") == 0) {
                float metallic = atof(tokens[1].c_str());
                newMaterial.metallic = metallic;
            }
            else if (strcmp(tokens[0].c_str(), "SPEC") == 0) {
                float spec = atof(tokens[1].c_str());
                newMaterial.specExponent = spec;
            }
            else if (strcmp(tokens[0].c_str(), "ASYM_TYPE") == 0) {
                if (strcmp(tokens[1].c_str(), "conductor")==0)
                    newMaterial.asymmicrofacet.type = conductor;
                else if(strcmp(tokens[1].c_str(), "dielectric") == 0)
                    newMaterial.asymmicrofacet.type = dielectric;
            }
            else if (strcmp(tokens[0].c_str(), "ASYM_ALPHA_X_A") == 0) {
                float val = atof(tokens[1].c_str());
                newMaterial.asymmicrofacet.alphaXA = val;
            }
            else if (strcmp(tokens[0].c_str(), "ASYM_ALPHA_Y_A") == 0) {
                float val = atof(tokens[1].c_str());
                newMaterial.asymmicrofacet.alphaYA = val;
            }
            else if (strcmp(tokens[0].c_str(), "ASYM_ALPHA_X_B") == 0) {
                float val = atof(tokens[1].c_str());
                newMaterial.asymmicrofacet.alphaXB = val;
            }
            else if (strcmp(tokens[0].c_str(), "ASYM_ALPHA_Y_B") == 0) {
                float val = atof(tokens[1].c_str());
                newMaterial.asymmicrofacet.alphaYB = val;
            }
            else if (strcmp(tokens[0].c_str(), "ASYM_ALPHA_ZS") == 0) {
                float val = atof(tokens[1].c_str());
                newMaterial.asymmicrofacet.zs = val;
            }
            else if (strcmp(tokens[0].c_str(), "ASYM_ALBEDO") == 0) {
                glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.asymmicrofacet.albedo = color;
            }*/
        }
        LoadMaterialJobs.emplace_back(type, params);
        //materials.push_back(Material::create(type, params, gpuAlloc));
        return 1;
    }
}

void Scene::buildBVH()
{
    for (int i=0;i<objects.size();i++)
    {
        const Object& obj = objects[i];
        if (obj.type == TRIANGLE_MESH)
        {
            int meshId = obj.meshId;
            const TriangleMeshData& mesh = m_triangleMeshes[meshId];
            for (int j = 0; j != mesh.m_triangles.size(); j++)
            {
                primitives.emplace_back(obj, i, j, mesh.m_triangles.data(), mesh.m_vertices.data());
            }
            
        }
        else
        {
            primitives.emplace_back(obj, i);
        }
    }
    bvhroot = buildBVHTreeRecursiveSAH(primitives, 0, primitives.size(), &bvhTreeSize);
    assert(checkBVHTreeFull(bvhroot));
}


void Scene::buildStacklessBVH()
{
#if MTBVH
    compactBVHTreeToMTBVH(MTBVHArray, bvhroot, bvhTreeSize);
#else
    recursiveCompactBVHTreeForStacklessTraverse(bvhArray, bvhroot);
#endif
}




//from pbrt-v4
void InitAliasTable(const std::vector<float>& w, std::vector<AliasBin>& bins)
{
    bins.resize(w.size());
    double sum = std::accumulate(w.begin(), w.end(), 0.0);
    for (size_t i = 0; i < w.size(); i++)
    {
        bins[i].p = w[i] / sum;
    }
    struct Work {
        float pScaled;
        size_t idx;
        Work(float ps, size_t i):pScaled(ps),idx(i){}
    };
    std::queue<Work> qUnder, qOver;
    for (size_t i = 0; i < bins.size(); i++)
    {
        float pScaled = bins[i].p * bins.size();
        if (pScaled < 1.0)
            qUnder.emplace(pScaled, i);
        else
            qOver.emplace(pScaled, i);
    }
    while (!qUnder.empty() && !qOver.empty())
    {
        auto under = qUnder.front(); qUnder.pop();
        auto over = qOver.front(); qOver.pop();
        bins[under.idx].q = under.pScaled;
        bins[under.idx].alias = over.idx;
        float pEx = over.pScaled - (1.0 - under.pScaled);
        if (pEx < 1.0)
            qOver.emplace(pEx, over.idx);
        else
            qUnder.emplace(pEx, over.idx);
    }
    while (!qOver.empty()) {
        Work over = qOver.back();
        qOver.pop();
        bins[over.idx].q = 1;
        bins[over.idx].alias = -1;
    }
    while (!qUnder.empty()) {
        Work under = qUnder.back();
        qUnder.pop();
        bins[under.idx].q = 1;
        bins[under.idx].alias = -1;
    }
}