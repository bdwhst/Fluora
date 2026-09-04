#include "scene_loader.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <tuple>
#include <unordered_map>

#include <json.hpp>

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

std::string sceneDirOf(const std::string& path)
{
    if (auto slash = path.find_last_of("/\\"); slash != std::string::npos)
        return path.substr(0, slash + 1);
    return "./";
}

std::string lowerExt(const std::string& path)
{
    auto dot = path.find_last_of('.');
    std::string ext = dot == std::string::npos ? "" : path.substr(dot);
    for (auto& c : ext)
        c = (char)tolower((unsigned char)c);
    return ext;
}

CoreObject makeAnalytic(CoreGeomType type, int materialId, const glm::mat4& t)
{
    CoreObject o;
    o.geomType = type;
    o.materialId = materialId;
    o.transform = hostStore4x4(t);
    o.invTransform = hostStore4x4(glm::inverse(t));
    o.invTranspose = hostStore4x4(glm::inverseTranspose(t));
    return o;
}

// Texture paths dedupe to one heap slot (MTL files commonly bind the same
// atlas to every material).
struct TextureRegistry {
    CoreScene& out;
    std::unordered_map<std::string, uint32_t> pathToTexIdx;
    uint32_t index(const std::string& p)
    {
        auto it = pathToTexIdx.find(p);
        if (it == pathToTexIdx.end()) {
            it = pathToTexIdx.emplace(p, (uint32_t)out.texturePaths.size()).first;
            out.texturePaths.push_back(p);
        }
        return it->second;
    }
};

// Shared tail of both loaders: reference checks, then the BVH.
bool finishScene(CoreScene& out, std::string& err)
{
    // An env map alone is something to render: a .json scene whose only object
    // is a medium boundary (dropped until media land) still shows its sky,
    // rather than failing to open.
    if (out.objects.empty() && out.tris.empty() && out.envMapPath.empty()) {
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

bool loadScene(const std::string& path, CoreScene& out, std::string& err)
{
    std::string ext = lowerExt(path);
    if (ext == ".json")
        return loadJsonScene(path, out, err);
    if (ext == ".txt")
        return loadTxtScene(path, out, err);
    err = "unknown scene extension '" + ext + "' (expected .txt or .json)";
    return false;
}

// ---- .txt -----------------------------------------------------------------

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

    const std::string sceneDir = sceneDirOf(path);
    TextureRegistry textures{ out };

    bool meshLoadFailed = false;
    auto flushObject = [&]() {
        if (!curObj.active)
            return;
        if (curObj.geometry == "cube" || curObj.geometry == "sphere") {
            out.objects.push_back(makeAnalytic(
                curObj.geometry == "cube" ? CORE_GEOM_CUBE : CORE_GEOM_SPHERE, curObj.materialId,
                buildTransform(curObj.trans, curObj.rot, curObj.scale)));
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
                                                     : textures.index(mm.diffuseTexPath);
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
    return finishScene(out, err);
}

// ---- .json ----------------------------------------------------------------
// Mirrors Scene::loadJSON key for key. Paths (meshes, textures, env map, VDB
// grids) are joined with the scene directory like the .txt loader; the
// checked-in scenes write them as ../scenes/..., which resolves the same
// from the repo root or a first-level subdirectory.

namespace {

using json = nlohmann::json;

glm::vec3 jsonVec3(const json& a)
{
    return glm::vec3(a.at(0).get<float>(), a.at(1).get<float>(), a.at(2).get<float>());
}

// TRANS / ROTAT / SCALE, each optional (identity when absent).
glm::mat4 jsonTransform(const json& o)
{
    glm::vec3 t(0.0f), r(0.0f), s(1.0f);
    if (o.contains("TRANS"))
        t = jsonVec3(o["TRANS"]);
    if (o.contains("ROTAT"))
        r = jsonVec3(o["ROTAT"]);
    if (o.contains("SCALE"))
        s = jsonVec3(o["SCALE"]);
    return buildTransform(t, r, s);
}

// {"TYPE": "...", "VALUE": ...} parameter blocks.
std::string jsonParamType(const json& p)
{
    return p.contains("TYPE") ? p["TYPE"].get<std::string>() : std::string();
}

} // namespace

bool loadJsonScene(const std::string& path, CoreScene& out, std::string& err)
{
    std::ifstream file(path);
    if (!file) {
        err = "cannot open scene file: " + path;
        return false;
    }
    const std::string sceneDir = sceneDirOf(path);
    TextureRegistry textures{ out };

    try {
        json data = json::parse(file);

        // Camera
        const json& cam = data.at("Camera");
        out.camera.width = cam.at("RES").at(0).get<int>();
        out.camera.height = cam.at("RES").at(1).get<int>();
        out.camera.fovyDeg = cam.at("FOVY").get<float>();
        out.camera.iterations = cam.at("ITERATIONS").get<int>();
        out.camera.maxDepth = cam.at("DEPTH").get<int>();
        out.camera.outputName = cam.at("FILE").get<std::string>();
        out.camera.eye = jsonVec3(cam.at("EYE"));
        out.camera.lookAt = jsonVec3(cam.at("LOOKAT"));
        out.camera.up = jsonVec3(cam.at("UP"));
        if (cam.contains("LENS_RADIUS"))
            out.camera.lensRadius = cam["LENS_RADIUS"].get<float>();
        if (cam.contains("FOCAL_LEN"))
            out.camera.focalLength = cam["FOCAL_LEN"].get<float>();

        // Background
        if (data.contains("Background")) {
            const json& bg = data["Background"];
            if (jsonParamType(bg) == "skybox" && bg.contains("PATH"))
                out.envMapPath = sceneDir + bg["PATH"].get<std::string>();
            if (bg.contains("SCALE"))
                out.envScale = bg["SCALE"].get<float>();
            if (bg.contains("MAXRGB"))
                out.envMaxRadiance = jsonVec3(bg["MAXRGB"]);
        }

        // Materials (named; ids in key order, which is how they are
        // referenced by objects, so the order itself does not matter)
        std::unordered_map<std::string, int> materialIds;
        if (data.contains("Materials")) {
            for (const auto& [name, val] : data["Materials"].items()) {
                CoreMaterial m;
                std::string type = val.at("TYPE").get<std::string>();
                if (type == "diffuse") {
                    m.type = CoreMaterialType::Diffuse;
                    if (val.contains("REFL")) {
                        const json& refl = val["REFL"];
                        std::string rt = jsonParamType(refl);
                        if (rt == "RGB" || rt == "rgb")
                            m.rgb = jsonVec3(refl.at("VALUE"));
                        else if (rt == "TEX")
                            m.texIdx = textures.index(sceneDir + refl.at("VALUE").get<std::string>());
                        else
                            std::cout << "core: material '" << name << "': unknown REFL type '"
                                      << rt << "'\n";
                    } else if (val.contains("RGB")) {
                        m.rgb = jsonVec3(val["RGB"]);
                    }
                } else if (type == "emissive") {
                    m.type = CoreMaterialType::Emissive;
                    m.rgb = jsonVec3(val.at("RGB"));
                    m.emittance = val.contains("EMITTANCE") ? val["EMITTANCE"].get<float>() : 1.0f;
                } else if (type == "dielectric") {
                    m.type = CoreMaterialType::Dielectric;
                    if (val.contains("ETA")) {
                        const json& eta = val["ETA"];
                        std::string et = jsonParamType(eta);
                        if (et == "const")
                            m.ior = eta.at("VALUE").get<float>();
                        else if (et == "named")
                            m.etaNamed = eta.at("VALUE").get<std::string>();
                    }
                } else if (type == "conductor") {
                    m.type = CoreMaterialType::Conductor;
                    if (val.contains("ETA") && jsonParamType(val["ETA"]) == "named")
                        m.etaNamed = val["ETA"].at("VALUE").get<std::string>();
                    if (val.contains("K") && jsonParamType(val["K"]) == "named")
                        m.kNamed = val["K"].at("VALUE").get<std::string>();
                    if (val.contains("RGB"))
                        m.rgb = jsonVec3(val["RGB"]);
                    if (val.contains("ROUGHNESS"))
                        m.roughness = val["ROUGHNESS"].get<float>();
                } else {
                    std::cout << "core: material '" << name << "' type '" << type
                              << "' unsupported, using diffuse\n";
                }
                if (val.contains("NORMAL_MAP"))
                    m.normalMapPath = sceneDir + val["NORMAL_MAP"].get<std::string>();
                materialIds[name] = (int)out.materials.size();
                out.materials.push_back(std::move(m));
            }
        }

        // Media
        std::unordered_map<std::string, int> mediumIds;
        if (data.contains("Media")) {
            for (const auto& [name, val] : data["Media"].items()) {
                CoreMedium med;
                med.name = name;
                std::string type = val.at("TYPE").get<std::string>();
                if (type == "nanovdb") {
                    med.type = CoreMediumType::NanoVdb;
                    med.vdbPath = sceneDir + val.at("PATH").get<std::string>();
                    if (val.contains("TEMPSCALE"))
                        med.temperatureScale = val["TEMPSCALE"].get<float>();
                    if (val.contains("TEMPOFFSET"))
                        med.temperatureOffset = val["TEMPOFFSET"].get<float>();
                } else if (type == "homogeneous") {
                    med.type = CoreMediumType::Homogeneous;
                } else {
                    err = "medium '" + name + "': unknown TYPE '" + type + "'";
                    return false;
                }
                if (val.contains("LESCALE"))
                    med.leScale = val["LESCALE"].get<float>();
                med.sigmaA = jsonVec3(val.at("SIGMA_A").at("VALUE"));
                med.sigmaS = jsonVec3(val.at("SIGMA_S").at("VALUE"));
                if (val.contains("SIGMA_SCALE"))
                    med.sigmaScale = val["SIGMA_SCALE"].get<float>();
                if (val.contains("G"))
                    med.g = val["G"].get<float>();
                med.worldFromMedium = jsonTransform(val);
                mediumIds[name] = (int)out.media.size();
                out.media.push_back(std::move(med));
            }
        }
        auto mediumId = [&](const std::string& name) {
            if (name.empty())
                return -1;
            auto it = mediumIds.find(name);
            if (it == mediumIds.end()) {
                std::cout << "core: unknown medium '" << name << "', treating as vacuum\n";
                return -1;
            }
            return it->second;
        };
        if (cam.contains("MEDIUM"))
            out.camera.mediumId = mediumId(cam["MEDIUM"].get<std::string>());

        // Medium interfaces: name -> (inside, outside) medium ids
        std::unordered_map<std::string, std::pair<int, int>> interfaces;
        if (data.contains("MediumInterfaces")) {
            for (const auto& [name, val] : data["MediumInterfaces"].items()) {
                interfaces[name] = { mediumId(val.at("INSIDE").get<std::string>()),
                                     mediumId(val.at("OUTSIDE").get<std::string>()) };
            }
        }

        // Objects. A medium interface rides on the material (triangles carry
        // one id), so a material used with an interface is cloned per
        // (material, inside, outside), and interface-only objects get a
        // pass-through Interface material per (inside, outside).
        std::map<std::tuple<int, int, int>, int> interfaceMaterials;
        auto objectMaterial = [&](const json& obj, std::string& e) {
            int matId = -1;
            if (obj.contains("MATERIAL")) {
                std::string mname = obj["MATERIAL"].get<std::string>();
                auto it = materialIds.find(mname);
                if (it == materialIds.end()) {
                    e = "object references unknown material '" + mname + "'";
                    return -1;
                }
                matId = it->second;
            }
            std::pair<int, int> iface{ -1, -1 };
            bool hasIface = false;
            if (obj.contains("MEDIUM_INTERFACE")) {
                std::string iname = obj["MEDIUM_INTERFACE"].get<std::string>();
                auto it = interfaces.find(iname);
                if (it == interfaces.end()) {
                    e = "object references unknown medium interface '" + iname + "'";
                    return -1;
                }
                iface = it->second;
                hasIface = true;
            }
            if (!hasIface) {
                if (matId < 0)
                    e = "object has neither MATERIAL nor MEDIUM_INTERFACE";
                return matId;
            }
            auto key = std::make_tuple(matId, iface.first, iface.second);
            auto it = interfaceMaterials.find(key);
            if (it != interfaceMaterials.end())
                return it->second;
            CoreMaterial m = matId >= 0 ? out.materials[matId] : CoreMaterial{};
            if (matId < 0)
                m.type = CoreMaterialType::Interface;
            m.mediumIn = iface.first;
            m.mediumOut = iface.second;
            int id = (int)out.materials.size();
            out.materials.push_back(std::move(m));
            interfaceMaterials[key] = id;
            return id;
        };

        int skippedInterfaces = 0;
        for (const json& obj : data.at("Objects")) {
            std::string type = obj.at("TYPE").get<std::string>();
            // Unsupported types are skipped before the material is resolved: an
            // unported type need carry no MATERIAL of its own, and resolving it
            // first would turn "skip with a warning" into a failed scene load.
            if (type != "geometry_cube" && type != "geometry_sphere" && type != "model_inline"
                && type != "model_ply") {
                std::cout << "core: skipping unsupported object type '" << type << "'\n";
                continue;
            }
            std::string e;
            int matId = objectMaterial(obj, e);
            if (!e.empty()) {
                err = e;
                return false;
            }
            // An interface-only object is a medium boundary with no surface, and
            // media are not rendered yet. Its geometry cannot simply be left in:
            // shadow rays test geometry alone, with no material (sceneOccluded),
            // so the boundary would cast a hard shadow that no BSDF-side
            // pass-through undoes. Drop it; the material and its media stay in
            // the scene for the media step to pick up.
            if (matId >= 0 && out.materials[matId].type == CoreMaterialType::Interface) {
                skippedInterfaces++;
                continue;
            }
            glm::mat4 t = jsonTransform(obj);
            if (type == "geometry_cube" || type == "geometry_sphere") {
                out.objects.push_back(makeAnalytic(
                    type == "geometry_cube" ? CORE_GEOM_CUBE : CORE_GEOM_SPHERE, matId, t));
            } else if (type == "model_inline") {
                std::vector<float> xyz = obj.at("VERTICES").get<std::vector<float>>();
                std::vector<uint32_t> idx = obj.at("INDICES").get<std::vector<uint32_t>>();
                if (obj.contains("NORMALS"))
                    std::cout << "core: model_inline NORMALS not supported, ignoring\n";
                if (!appendInlineMesh(xyz, idx, t, (uint32_t)matId, out.positions, out.normals,
                                      out.uvs, out.tris, err))
                    return false;
            } else if (type == "model_ply") {
                // Dispatch on the real extension: scenes label OBJ files
                // model_ply too (bunny.json).
                std::string mpath = sceneDir + obj.at("PATH").get<std::string>();
                bool ok;
                if (lowerExt(mpath) == ".obj") {
                    std::vector<MeshMaterial> unused;
                    ok = loadObjMesh(mpath, t, matId, 0, /*useVertexNormal=*/true, out.positions,
                                     out.normals, out.uvs, out.tris, unused);
                } else {
                    ok = loadPlyMesh(mpath, t, (uint32_t)matId, out.positions, out.normals,
                                     out.uvs, out.tris);
                }
                if (!ok) {
                    err = "failed to load mesh " + mpath;
                    return false;
                }
            }
        }
        if (skippedInterfaces)
            std::cout << "core: " << skippedInterfaces
                      << " medium-interface object(s) skipped (no surface to render yet)\n";
    } catch (const std::exception& e) {
        err = std::string("json scene: ") + e.what();
        return false;
    }
    if (!out.media.empty())
        std::cout << "core: " << out.media.size()
                  << " media parsed (volumes not rendered yet; boundaries not rendered)\n";
    return finishScene(out, err);
}
