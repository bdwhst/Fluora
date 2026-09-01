#include "scene_loader.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "host_math.h"
#include "mesh_loader.h"

namespace {

constexpr float kPi = 3.14159265358979323846f;

// utilityCore::buildTransformationMatrix, statement for statement, so baked
// transforms are float-identical across backend hosts.
glm::mat4 buildTransform(glm::vec3 translation, glm::vec3 rotationDeg, glm::vec3 scale)
{
    glm::mat4 translationMat = glm::translate(glm::mat4(), translation);
    glm::mat4 rotationMat = glm::rotate(glm::mat4(), rotationDeg.x * kPi / 180, glm::vec3(1, 0, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(), rotationDeg.y * kPi / 180, glm::vec3(0, 1, 0));
    rotationMat = rotationMat * glm::rotate(glm::mat4(), rotationDeg.z * kPi / 180, glm::vec3(0, 0, 1));
    glm::mat4 scaleMat = glm::scale(glm::mat4(), scale);
    return translationMat * rotationMat * scaleMat;
}

glm::vec3 readVec3(std::istringstream& ss)
{
    float x = 0, y = 0, z = 0;
    ss >> x >> y >> z;
    return glm::vec3(x, y, z);
}

struct PendingObject {
    std::string geometry;
    std::string meshPath;
    bool useVertexNormal = false;
    int materialId = -1;
    glm::vec3 trans { 0, 0, 0 };
    glm::vec3 rot { 0, 0, 0 };
    glm::vec3 scale { 1, 1, 1 };
    bool active = false;
};

} // namespace

bool loadTxtScene(const std::string& path, CoreScene& out, std::string& err)
{
    std::ifstream file(path);
    if (!file) {
        err = "cannot open scene file: " + path;
        return false;
    }

    enum class Mode { None, Material, Camera, Object };
    Mode mode = Mode::None;
    CoreMaterial curMat;
    PendingObject curObj;

    std::string sceneDir = "./";
    if (auto slash = path.find_last_of('/'); slash != std::string::npos)
        sceneDir = path.substr(0, slash + 1);

    // Texture paths dedupe to one heap slot (MTL files commonly bind the same
    // atlas to every material).
    std::unordered_map<std::string, uint32_t> pathToTexIdx;
    auto textureIndex = [&](const std::string& p) {
        auto it = pathToTexIdx.find(p);
        if (it == pathToTexIdx.end()) {
            it = pathToTexIdx.emplace(p, (uint32_t)out.texturePaths.size()).first;
            out.texturePaths.push_back(p);
        }
        return it->second;
    };

    bool meshLoadFailed = false;
    auto flushObject = [&]() {
        if (!curObj.active)
            return;
        if (curObj.geometry == "cube" || curObj.geometry == "sphere") {
            glm::mat4 t = buildTransform(curObj.trans, curObj.rot, curObj.scale);
            CoreObject o;
            o.geomType = curObj.geometry == "cube" ? CORE_GEOM_CUBE : CORE_GEOM_SPHERE;
            o.materialId = curObj.materialId;
            o.transform = hostStore4x4(t);
            o.invTransform = hostStore4x4(glm::inverse(t));
            o.invTranspose = hostStore4x4(glm::inverseTranspose(t));
            out.objects.push_back(o);
        } else if (curObj.geometry == "mesh") {
            // Model paths in scene files are relative to the scene directory.
            // material -1 = use the OBJ's MTL materials (Scene::loadModel
            // convention): they append to the scene material list as textured
            // diffuse, with each texture path registered for heap upload.
            std::vector<MeshMaterial> mtlMats;
            if (!loadObjMesh(sceneDir + curObj.meshPath,
                             buildTransform(curObj.trans, curObj.rot, curObj.scale),
                             curObj.materialId, (uint32_t)out.materials.size(),
                             curObj.useVertexNormal,
                             out.positions, out.normals, out.uvs, out.tris, mtlMats)) {
                err = "failed to load mesh " + sceneDir + curObj.meshPath;
                meshLoadFailed = true;
                curObj = PendingObject{};
                return;
            }
            for (const auto& mm : mtlMats) {
                CoreMaterial m;
                m.type = CoreMaterialType::Diffuse;
                m.rgb = mm.kd;
                m.texIdx = mm.diffuseTexPath.empty() ? kCoreTexNone
                                                     : textureIndex(mm.diffuseTexPath);
                out.materials.push_back(std::move(m));
            }
        } else {
            std::cout << "core: skipping unsupported geometry '" << curObj.geometry << "'\n";
        }
        curObj = PendingObject{};
    };
    auto flushMaterial = [&]() {
        if (mode == Mode::Material)
            out.materials.push_back(curMat);
    };

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string tok;
        if (!(ss >> tok) || tok.rfind("//", 0) == 0)
            continue;

        if (tok == "MATERIAL") {
            flushMaterial();
            flushObject();
            mode = Mode::Material;
            curMat = CoreMaterial{};
        } else if (tok == "CAMERA") {
            flushMaterial();
            flushObject();
            mode = Mode::Camera;
        } else if (tok == "OBJECT") {
            flushMaterial();
            flushObject();
            mode = Mode::Object;
            curObj.active = true;
        } else if (tok == "SKYBOX") {
            // Like Scene::loadSkybox: the path is the whole next line,
            // relative to the scene directory.
            flushMaterial();
            flushObject();
            mode = Mode::None;
            std::string pathLine;
            if (std::getline(file, pathLine)) {
                while (!pathLine.empty() && (pathLine.back() == '\r' || pathLine.back() == ' '))
                    pathLine.pop_back();
                if (!pathLine.empty())
                    out.envMapPath = sceneDir + pathLine;
            }
        }
        // material keys
        else if (mode == Mode::Material && tok == "TYPE") {
            std::string type;
            ss >> type;
            if (type == "diffuse")
                curMat.type = CoreMaterialType::Diffuse;
            else if (type == "emitting")
                curMat.type = CoreMaterialType::Emissive;
            else if (type == "frenselSpecular")
                curMat.type = CoreMaterialType::Dielectric;
            else if (type == "microfacet" || type == "conductor")
                curMat.type = CoreMaterialType::Conductor;
            else {
                std::cout << "core: material type '" << type << "' unsupported, using diffuse\n";
                curMat.type = CoreMaterialType::Diffuse;
            }
        } else if (mode == Mode::Material && tok == "RGB") {
            curMat.rgb = readVec3(ss);
        } else if (mode == Mode::Material && tok == "REFRIOR") {
            ss >> curMat.ior;
            if (curMat.ior <= 0.0f)
                curMat.ior = 1.5f;
        } else if (mode == Mode::Material
                   && (tok == "REFRIOR_NAMED" || tok == "REFRIOR_REAL_NAMED")) {
            ss >> curMat.etaNamed;
        } else if (mode == Mode::Material && tok == "REFRIOR_IMAG_NAMED") {
            ss >> curMat.kNamed;
        } else if (mode == Mode::Material && tok == "REFRIOR_RGB") {
            curMat.etaRgb = readVec3(ss);
            curMat.hasEtaRgb = true;
        } else if (mode == Mode::Material && tok == "EMITTANCE") {
            ss >> curMat.emittance;
        } else if (mode == Mode::Material && tok == "ROUGHNESS") {
            ss >> curMat.roughness;
        }
        // camera keys
        else if (mode == Mode::Camera && tok == "RES") {
            ss >> out.camera.width >> out.camera.height;
        } else if (mode == Mode::Camera && tok == "FOVY") {
            ss >> out.camera.fovyDeg;
        } else if (mode == Mode::Camera && tok == "ITERATIONS") {
            ss >> out.camera.iterations;
        } else if (mode == Mode::Camera && tok == "DEPTH") {
            ss >> out.camera.maxDepth;
        } else if (mode == Mode::Camera && tok == "FILE") {
            ss >> out.camera.outputName;
        } else if (mode == Mode::Camera && tok == "EYE") {
            out.camera.eye = readVec3(ss);
        } else if (mode == Mode::Camera && tok == "LOOKAT") {
            out.camera.lookAt = readVec3(ss);
        } else if (mode == Mode::Camera && tok == "UP") {
            out.camera.up = readVec3(ss);
        }
        // object keys
        else if (mode == Mode::Object && tok == "geometry") {
            ss >> curObj.geometry;
        } else if (mode == Mode::Object && tok == "model") {
            std::string subtype;
            ss >> subtype >> curObj.meshPath;
            curObj.useVertexNormal = subtype == "vnormal";
            curObj.geometry = "mesh";
        } else if (mode == Mode::Object && tok == "material") {
            ss >> curObj.materialId;
        } else if (mode == Mode::Object && tok == "TRANS") {
            curObj.trans = readVec3(ss);
        } else if (mode == Mode::Object && tok == "ROTAT") {
            curObj.rot = readVec3(ss);
        } else if (mode == Mode::Object && tok == "SCALE") {
            curObj.scale = readVec3(ss);
        }
    }
    flushMaterial();
    flushObject();

    if (meshLoadFailed)
        return false;  // err set at the failure site
    if (out.objects.empty() && out.tris.empty()) {
        err = "scene has no renderable geometry";
        return false;
    }
    for (const auto& o : out.objects) {
        if (o.materialId < 0 || o.materialId >= (int)out.materials.size()) {
            err = "object references material out of range";
            return false;
        }
    }
    // Mesh triangles carry material ids in uint4.w (the scene's "material N"
    // line, or MTL-derived ids); kernels index materials[] with them
    // unchecked, so a bad id must die here, not on the GPU.
    for (const auto& tri : out.tris) {
        if (tri.w >= out.materials.size()) {
            err = "mesh triangle references material out of range";
            return false;
        }
    }
    out.bvh = buildThreadedBvh6(out.positions, out.tris);
    return true;
}
