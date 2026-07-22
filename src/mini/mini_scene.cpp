#include "mini_scene.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {

constexpr float kPi = 3.14159265358979323846f;

simd_float4x4 makeTranslate(simd_float3 t)
{
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[3] = simd_make_float4(t, 1.0f);
    return m;
}

simd_float4x4 makeScale(simd_float3 s)
{
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[0][0] = s.x;
    m.columns[1][1] = s.y;
    m.columns[2][2] = s.z;
    return m;
}

simd_float4x4 makeRotX(float a)
{
    float c = std::cos(a), s = std::sin(a);
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[1] = simd_make_float4(0.0f, c, s, 0.0f);
    m.columns[2] = simd_make_float4(0.0f, -s, c, 0.0f);
    return m;
}

simd_float4x4 makeRotY(float a)
{
    float c = std::cos(a), s = std::sin(a);
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[0] = simd_make_float4(c, 0.0f, -s, 0.0f);
    m.columns[2] = simd_make_float4(s, 0.0f, c, 0.0f);
    return m;
}

simd_float4x4 makeRotZ(float a)
{
    float c = std::cos(a), s = std::sin(a);
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[0] = simd_make_float4(c, s, 0.0f, 0.0f);
    m.columns[1] = simd_make_float4(-s, c, 0.0f, 0.0f);
    return m;
}

// Same composition as utilityCore::buildTransformationMatrix: T * Rx * Ry * Rz * S,
// rotations in degrees.
simd_float4x4 buildTransform(simd_float3 t, simd_float3 rDeg, simd_float3 s)
{
    simd_float4x4 m = makeTranslate(t);
    m = simd_mul(m, makeRotX(rDeg.x * kPi / 180.0f));
    m = simd_mul(m, makeRotY(rDeg.y * kPi / 180.0f));
    m = simd_mul(m, makeRotZ(rDeg.z * kPi / 180.0f));
    m = simd_mul(m, makeScale(s));
    return m;
}

simd_float3 readVec3(std::istringstream& ss)
{
    float x = 0, y = 0, z = 0;
    ss >> x >> y >> z;
    return simd_make_float3(x, y, z);
}

struct PendingObject {
    std::string geometry;
    int materialId = -1;
    simd_float3 trans = { 0, 0, 0 };
    simd_float3 rot = { 0, 0, 0 };
    simd_float3 scale = { 1, 1, 1 };
    bool active = false;
};

} // namespace

bool miniLoadScene(const std::string& path, MiniScene& out, std::string& err)
{
    std::ifstream file(path);
    if (!file) {
        err = "cannot open scene file: " + path;
        return false;
    }

    enum class Mode { None, Material, Camera, Object };
    Mode mode = Mode::None;
    MiniMaterial curMat = {};
    PendingObject curObj;

    auto flushObject = [&]() {
        if (!curObj.active)
            return;
        if (curObj.geometry == "cube" || curObj.geometry == "sphere") {
            MiniObject o = {};
            o.geomType = curObj.geometry == "cube" ? MINI_GEOM_CUBE : MINI_GEOM_SPHERE;
            o.materialId = curObj.materialId;
            o.transform = buildTransform(curObj.trans, curObj.rot, curObj.scale);
            o.invTransform = simd_inverse(o.transform);
            o.invTranspose = simd_transpose(o.invTransform);
            out.objects.push_back(o);
        } else {
            std::cout << "mini: skipping unsupported geometry '" << curObj.geometry << "'\n";
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
            curMat = MiniMaterial{};
            curMat.rgb = simd_make_float3(0.5f, 0.5f, 0.5f);
            curMat.ior = 1.5f;
        } else if (tok == "CAMERA") {
            flushMaterial();
            flushObject();
            mode = Mode::Camera;
        } else if (tok == "OBJECT") {
            flushMaterial();
            flushObject();
            mode = Mode::Object;
            curObj.active = true;
        }
        // material keys
        else if (mode == Mode::Material && tok == "TYPE") {
            std::string type;
            ss >> type;
            if (type == "diffuse")
                curMat.type = MINI_MAT_DIFFUSE;
            else if (type == "emitting")
                curMat.type = MINI_MAT_EMITTING;
            else if (type == "frenselSpecular")
                curMat.type = MINI_MAT_GLASS;
            else if (type == "microfacet")
                curMat.type = MINI_MAT_MIRROR;  // M1: roughness ignored
            else {
                std::cout << "mini: material type '" << type << "' unsupported, using diffuse\n";
                curMat.type = MINI_MAT_DIFFUSE;
            }
        } else if (mode == Mode::Material && tok == "RGB") {
            curMat.rgb = readVec3(ss);
        } else if (mode == Mode::Material && tok == "REFRIOR") {
            ss >> curMat.ior;
            if (curMat.ior <= 0.0f)
                curMat.ior = 1.5f;
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

    if (out.objects.empty()) {
        err = "scene has no cube/sphere objects that FluoraMini can render";
        return false;
    }
    for (const auto& o : out.objects) {
        if (o.materialId < 0 || o.materialId >= (int)out.materials.size()) {
            err = "object references material out of range";
            return false;
        }
    }
    return true;
}
