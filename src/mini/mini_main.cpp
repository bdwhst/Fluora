// FluoraMini: Cornell-box vertical slice on the Metal RHI backend (M1 in
// docs/metal-rhi-design.md). Loads a Fluora .txt or .json scene, path-traces it
// via rhi:: with a live preview window, and writes a PNG. The preview is
// interactive: an ImGui overlay (src/core/gui) shows render stats, a fly
// camera (WASD / drag / wheel) re-renders on move, and a dropdown hot-swaps
// scenes from the same directory. Headless mode (--no-preview) keeps the plain
// accumulate-N-samples loop and stays bitwise-identical to the reference PNGs.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <stb_image_write.h>

#include <glm/glm.hpp>

#include "../core/gui/gui.h"
#include "../core/host_math.h"
#include "../core/image_loader.h"
#include "../core/lights.h"
#include "../core/scene_loader.h"
#include "../core/spectra.h"
#include "../core/spectrum_shared.h"
#include "../core/tonemap_shared.h"
#include "../rhi/rhi.h"
#include "../rhi/primitives_shared.h"
#include "mini_shared.h"

namespace {

std::string readTextFile(const std::string& path)
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// All GPU resources that depend on the loaded scene: materials, analytic
// objects, the BVH/mesh buffers behind the ray-tracing seam, per-material
// textures, the environment map, and the spectral tables. Session-invariant
// resources (accumulation buffer, wavefront queues, present target, pipelines)
// live in main() and survive a scene swap. Rebuilt wholesale on scene switch;
// the old one's buffers free when this is reassigned (after a waitIdle()).
struct SceneGpu {
    std::unique_ptr<rhi::Buffer> matBuf, objBuf, normalBuf, uvBuf, spdBuf, r2sBuf;
    std::unique_ptr<rhi::Buffer> lightBuf, envDistBuf;
    std::unique_ptr<rhi::RayIntersector> intersector;
    std::array<rhi::Buffer*, 3> rt{ nullptr, nullptr, nullptr };
    std::vector<std::unique_ptr<rhi::Texture>> matTextures;
    std::unique_ptr<rhi::Texture> envTex;
    uint32_t envMapIdx = MINI_ENV_NONE;

    // Derived params fields.
    unsigned numObjects = 0;
    unsigned bvhNumNodes = 0;
    unsigned numLights = 0;
    unsigned envW = 0, envH = 0;
    int maxDepth = 8;
    float fovyDeg = 45.0f;
    size_t numTris = 0;

    // Camera defaults straight from the scene file.
    glm::vec3 eye{ 0, 0, 0 }, lookAt{ 0, 0, -1 }, up{ 0, 1, 0 };
    std::string outputName = "mini";
};

// CoreScene -> device PODs + GPU uploads. Spectra are rebuilt per scene because
// materials reference named eta/k spectra by offset into this table.
SceneGpu buildSceneGpu(rhi::Device& device, const CoreScene& scene)
{
    SceneGpu sg;
    sg.numObjects = (unsigned)scene.objects.size();
    sg.maxDepth = scene.camera.maxDepth;
    sg.fovyDeg = scene.camera.fovyDeg;
    sg.numTris = scene.tris.size();
    sg.eye = scene.camera.eye;
    sg.lookAt = scene.camera.lookAt;
    sg.up = scene.camera.up;
    sg.outputName = scene.camera.outputName;

    SpectralTables spectra;
    auto resolveSpd = [&](const std::string& name) {
        if (name.empty())
            return (uint32_t)SPD_NONE;
        uint32_t off = spectra.namedOffset(name);
        if (off == SPD_NONE)
            std::cout << "mini: unknown named spectrum '" << name
                      << "', falling back to RGB parameters\n";
        return off;
    };

    // Base-color textures first, so material texIdx can be remapped to real heap
    // indices (the heap grows across scene loads; per-scene 0-based indices no
    // longer equal heap slots after the first load). A 1x1 white fallback keeps
    // the mapping dense when a file fails to load.
    std::vector<uint32_t> texHeap;  // per scene.texturePaths entry -> heap index
    texHeap.reserve(scene.texturePaths.size());
    for (const auto& texPath : scene.texturePaths) {
        LdrImage img;
        if (!loadLdrImage(texPath, img)) {
            std::cout << "mini: failed to load texture " << texPath << ", using white\n";
            img.width = img.height = 1;
            img.rgba = { 255, 255, 255, 255 };
        }
        auto tex = device.createTexture(
            { img.width, img.height, rhi::TextureFormat::RGBA8Unorm, true, /*srgb=*/true,
              "basecolor" });
        tex->upload(img.rgba.data(), img.rgba.size());
        texHeap.push_back((uint32_t)tex->shaderHandle());
        sg.matTextures.push_back(std::move(tex));
    }
    if (!scene.texturePaths.empty())
        std::cout << "mini: " << scene.texturePaths.size() << " material textures\n";

    std::vector<MiniMaterial> materials;
    materials.reserve(scene.materials.size());
    for (const auto& m : scene.materials) {
        MiniMaterial mm = {};
        switch (m.type) {
        case CoreMaterialType::Diffuse: mm.type = MINI_MAT_DIFFUSE; break;
        case CoreMaterialType::Emissive: mm.type = MINI_MAT_EMITTING; break;
        case CoreMaterialType::Dielectric: mm.type = MINI_MAT_GLASS; break;
        case CoreMaterialType::Conductor: mm.type = MINI_MAT_CONDUCTOR; break;
        // Medium boundary without a surface. The loader emits no geometry for
        // these (they would block shadow rays), so this entry only keeps the
        // material indices dense; an index-matched dielectric is the closest
        // stand-in until the media step gives it a real transition.
        case CoreMaterialType::Interface: mm.type = MINI_MAT_GLASS; break;
        }
        mm.rgb = hostStore3(m.rgb);
        mm.emittance = m.emittance;
        mm.ior = m.type == CoreMaterialType::Interface ? 1.0f : m.ior;
        mm.roughness = m.roughness;
        mm.texIdx = m.texIdx == kCoreTexNone ? MINI_TEX_NONE : texHeap[m.texIdx];
        mm.etaSpd = resolveSpd(m.etaNamed);
        mm.kSpd = resolveSpd(m.kNamed);
        // Conductor Fresnel needs eta AND k; with only one, use reflectance
        // mode (both SPD_NONE).
        if (mm.type == MINI_MAT_CONDUCTOR && (mm.etaSpd == SPD_NONE || mm.kSpd == SPD_NONE))
            mm.etaSpd = mm.kSpd = SPD_NONE;
        materials.push_back(mm);
    }
    std::vector<MiniObject> objects;
    objects.reserve(scene.objects.size());
    for (const auto& o : scene.objects) {
        MiniObject mo = {};
        mo.geomType = o.geomType == CORE_GEOM_SPHERE ? MINI_GEOM_SPHERE : MINI_GEOM_CUBE;
        mo.materialId = o.materialId;
        mo.transform = o.transform;
        mo.invTransform = o.invTransform;
        mo.invTranspose = o.invTranspose;
        objects.push_back(mo);
    }

    sg.matBuf = device.createBuffer(
        { std::max<size_t>(materials.size(), 1) * sizeof(MiniMaterial),
          rhi::MemoryLocation::Shared, "materials" });
    std::memcpy(sg.matBuf->hostPtr(), materials.data(), materials.size() * sizeof(MiniMaterial));
    sg.objBuf = device.createBuffer(
        { std::max<size_t>(objects.size(), 1) * sizeof(MiniObject),
          rhi::MemoryLocation::Shared, "objects" });
    std::memcpy(sg.objBuf->hostPtr(), objects.data(), objects.size() * sizeof(MiniObject));

    // Mesh geometry behind the ray-tracing seam: the CPU-built BVH is handed to
    // the RayIntersector, whose buffers we slot-bind for rt_closest_hit.
    sg.intersector = device.createIntersector();
    rhi::AccelBuildInput accel;
    accel.nodes = scene.bvh.nodes.data();
    accel.nodeBytes = scene.bvh.nodes.size() * sizeof(RtBvhNode);
    accel.numNodesPerDir = scene.bvh.numNodesPerDir;
    accel.triangles = scene.tris.data();
    accel.triangleBytes = scene.tris.size() * sizeof(gpu_uint4);
    accel.positions = scene.positions.data();
    accel.positionBytes = scene.positions.size() * sizeof(gpu_storage3);
    sg.intersector->build(accel);
    sg.rt = sg.intersector->bindings();  // {nodes, tris, positions}
    sg.bvhNumNodes = sg.intersector->numNodes();

    // Mesh vertex attributes (unified indices with positions); dummy-sized when
    // the scene has no meshes (Metal rejects zero-length buffers).
    auto makeUpload = [&](const void* data, size_t bytes, const char* name) {
        auto buf = device.createBuffer(
            { std::max<size_t>(bytes, 16), rhi::MemoryLocation::Shared, name });
        if (data && bytes)
            std::memcpy(buf->hostPtr(), data, bytes);
        return buf;
    };
    sg.normalBuf = makeUpload(scene.normals.data(),
                              scene.normals.size() * sizeof(gpu_storage3), "normals");
    sg.uvBuf = makeUpload(scene.uvs.data(), scene.uvs.size() * sizeof(gpu_float2), "uvs");

    // Environment map: an RGBA32F texture in the bindless heap; kernels get the
    // heap index through params (invariant I-1: index, not handle).
    std::vector<float> envDist;
    if (!scene.envMapPath.empty()) {
        HdrImage envImg;
        if (loadHdrImage(scene.envMapPath, envImg)) {
            applyEnvScale(envImg, scene.envScale, scene.envMaxRadiance);
            sg.envTex = device.createTexture(
                { envImg.width, envImg.height, rhi::TextureFormat::RGBA32Float, true, false,
                  "envmap" });
            sg.envTex->upload(envImg.rgba.data(), envImg.rgba.size() * sizeof(float));
            sg.envMapIdx = (uint32_t)sg.envTex->shaderHandle();
            sg.envW = (unsigned)envImg.width;
            sg.envH = (unsigned)envImg.height;
            // Luminance distribution for env-map importance sampling (NEE).
            envDist = buildEnvDistribution(envImg, scene.envMaxRadiance);
            std::cout << "mini: environment map " << scene.envMapPath << " (" << envImg.width
                      << "x" << envImg.height << ")\n";
        } else {
            std::cout << "mini: failed to load environment map " << scene.envMapPath
                      << ", using black\n";
        }
    }
    sg.envDistBuf = makeUpload(envDist.data(), envDist.size() * sizeof(float), "envdist");

    // Light list for next-event estimation: emissive objects/triangles + env.
    std::vector<RtLight> lights = buildLightList(scene, sg.envMapIdx != MINI_ENV_NONE);
    sg.numLights = (unsigned)lights.size();
    sg.lightBuf = makeUpload(lights.data(), lights.size() * sizeof(RtLight), "lights");
    std::cout << "mini: " << lights.size() << " lights\n";

    // Spectral tables (invariant I-1: kernels get offsets, not pointers): the
    // dense-spectra buffer (CIE curves, D65, named eta/k) and the sRGB rgb2spec
    // coefficient table (zNodes then coeffs).
    sg.spdBuf = device.createBuffer(
        { spectra.buffer().size() * sizeof(float), rhi::MemoryLocation::Shared, "spd" });
    std::memcpy(sg.spdBuf->hostPtr(), spectra.buffer().data(),
                spectra.buffer().size() * sizeof(float));
    Rgb2SpecView r2sView = rgb2specSrgb();
    sg.r2sBuf = device.createBuffer(
        { (r2sView.zNodeCount + r2sView.coeffCount) * sizeof(float), rhi::MemoryLocation::Shared,
          "rgb2spec" });
    std::memcpy(sg.r2sBuf->hostPtr(), r2sView.zNodes, r2sView.zNodeCount * sizeof(float));
    std::memcpy((float*)sg.r2sBuf->hostPtr() + r2sView.zNodeCount, r2sView.coeffs,
                r2sView.coeffCount * sizeof(float));
    return sg;
}

// Camera basis exactly as scene.cpp builds it (quirks included: camRight =
// cross(view, up), camUp used raw). Headless renders use this so their PNGs
// stay bitwise-identical to the references.
void setCameraExact(MiniParams& p, const glm::vec3& eye, const glm::vec3& lookAt,
                    const glm::vec3& up)
{
    glm::vec3 view = glm::normalize(lookAt - eye);
    p.camPos = hostStore3(eye);
    p.camView = hostStore3(view);
    p.camUp = hostStore3(up);
    p.camRight = hostStore3(glm::normalize(glm::cross(view, up)));
}

// Camera basis from the interactive fly camera: a proper orthonormal frame.
// Matches setCameraExact for the horizontal-view default scenes (pitch 0 ->
// up == world up), so entering the preview does not jump the framing.
void setCameraFly(MiniParams& p, const gui::FlyCamera& cam)
{
    p.camPos = hostStore3(cam.eye);
    p.camView = hostStore3(cam.forward());
    p.camRight = hostStore3(cam.right());
    p.camUp = hostStore3(cam.up());
}

// pixelLength quirk replicated from scene.cpp: tan(fovy in degrees->radians)
// un-halved, so FOVY behaves as a half-angle.
void setFilmParams(MiniParams& p, const glm::mat3& filmM, int width, int height, float fovyDeg)
{
    // Row i of the film matrix = (M[0][i], M[1][i], M[2][i]) (glm is
    // column-major); kernels apply rgb = (dot(r0,xyz), ...).
    p.filmR0 = hostStore3(glm::vec3(filmM[0][0], filmM[1][0], filmM[2][0]));
    p.filmR1 = hostStore3(glm::vec3(filmM[0][1], filmM[1][1], filmM[2][1]));
    p.filmR2 = hostStore3(glm::vec3(filmM[0][2], filmM[1][2], filmM[2][2]));
    float yscaled = std::tan(fovyDeg * 3.14159265358979323846f / 180.0f);
    float xscaled = yscaled * width / (float)height;
    p.pixelLenX = 2.0f * xscaled / (float)width;
    p.pixelLenY = 2.0f * yscaled / (float)height;
    p.width = width;
    p.height = height;
}

// ACES-tonemap the accumulator (divided by the sample count) into an sRGB PNG,
// mirrored to match saveImage() in main.cpp so images compare with img/ refs.
bool savePng(const std::string& name, const float* acc, int width, int height, int samples)
{
    std::vector<unsigned char> pixels((size_t)width * height * 3);
    // Divide (don't multiply by a reciprocal): 1/N isn't exact and the ULP
    // drift flips quantization-boundary bytes vs the img/ reference convention.
    float div = samples > 0 ? (float)samples : 1.0f;  // 0-sample accum is all zeros
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t src = (size_t)y * width + x;
            size_t dst = (size_t)y * width + (width - 1 - x);
            gpu_float3 v = tonemap_aces(
                gpu_float3(acc[src * 4 + 0], acc[src * 4 + 1], acc[src * 4 + 2]) / div);
            for (int ch = 0; ch < 3; ch++)
                pixels[dst * 3 + ch] = (unsigned char)(v[ch] * 255.0f + 0.5f);
        }
    }
    return stbi_write_png(name.c_str(), width, height, 3, pixels.data(), width * 3) != 0;
}

} // namespace

int main(int argc, char** argv)
{
    // With no scene argument, open the Cornell box from the source tree so the
    // app launches standalone (the compile-time path works from any cwd).
#ifdef FLUORA_SCENES_DIR
    std::string defaultScene = std::string(FLUORA_SCENES_DIR) + "/cornell-sphere.txt";
#else
    std::string defaultScene = "scenes/cornell-sphere.txt";
#endif
    // A lone flag (e.g. "--no-preview") is not a scene path.
    bool haveScene = argc >= 2 && argv[1][0] != '-';
    std::string scenePath = haveScene ? argv[1] : defaultScene;
    int sppOverride = -1;
    std::string outOverride;
    std::string mode = "wavefront";
    bool preview = true;
    bool safeMath = false;
    for (int i = haveScene ? 2 : 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--spp" && i + 1 < argc)
            sppOverride = std::atoi(argv[++i]);
        else if (arg == "--out" && i + 1 < argc)
            outOverride = argv[++i];
        else if (arg == "--mode" && i + 1 < argc)
            mode = argv[++i];
        else if (arg == "--no-preview")
            preview = false;
        else if (arg == "--safe-math")
            safeMath = true;
    }
    if (mode != "wavefront" && mode != "mega") {
        std::cerr << "unknown --mode " << mode << "\n";
        return 1;
    }

    CoreScene scene;
    std::string err;
    if (!loadScene(scenePath, scene, err)) {
        std::cerr << "scene load failed: " << err << "\n";
        return 1;
    }
    // Render size follows the loaded scene: a swapped-in scene at a different
    // resolution reallocates the size-dependent session buffers and resizes
    // the window through presentTarget() (the window is never user-resizable).
    int width = scene.camera.width;
    int height = scene.camera.height;
    const int spp = sppOverride > 0 ? sppOverride : scene.camera.iterations;
    std::cout << "mini: " << scene.objects.size() << " objects, " << scene.materials.size()
              << " materials, " << width << "x" << height << ", " << spp << " spp, " << mode
              << " mode, " << rhi::backendName(rhi::kNativeBackend) << " backend"
              << (safeMath ? ", safe math" : "") << "\n";

    // Sibling .txt/.json scenes populate the preview's dropdown (sorted by name).
    std::vector<std::string> scenePaths, sceneNames;
    int selectedScene = 0;
    gui::scanSceneDirectory(scenePath, sceneNames, scenePaths, selectedScene);

    try {
        // Metal: runtime MSL compile of shared structs + RHI primitives + renderer
        // kernels, concatenated (see DeviceDesc::shaderSource in rhi.h). CUDA:
        // the same files are compiled by nvcc (mini_kernels.cu) and registered.
        rhi::DeviceDesc deviceDesc;
        // --safe-math: compile shaders without fast math so mega and wavefront
        // are bitwise identical (DeviceDesc::safeMath in rhi.h) — the mode the
        // cmp-based regression check runs in. Default fast math keeps full
        // speed but allows last-ulp divergence between the modes.
        deviceDesc.safeMath = safeMath;
        if (rhi::kNativeBackend == rhi::BackendKind::Metal)
            deviceDesc.shaderSource = readTextFile(std::string(RHI_SHADER_DIR) + "/gpu_portable.h")
                                + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/spectrum_shared.h")
                                + "\n" + readTextFile(std::string(MINI_SHADER_DIR) + "/mini_shared.h")
                                + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/accel_shared.h")
                                + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/bsdf_shared.h")
                                + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/tonemap_shared.h")
                                + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/envmap_shared.h")
                                + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/light_shared.h")
                                + "\n" + readTextFile(std::string(RHI_SHADER_DIR) + "/primitives_shared.h")
                                + "\n" + readTextFile(std::string(RHI_SHADER_DIR) + "/primitives_gpu.h")
                                + "\n" + readTextFile(std::string(RHI_SHADER_DIR) + "/raytrace_gpu.h")
                                + "\n" + readTextFile(std::string(RHI_SHADER_DIR) + "/texture_gpu.h")
                                + "\n" + readTextFile(std::string(MINI_SHADER_DIR) + "/pathtrace_gpu.h");
        auto device = rhi::createDevice(rhi::kNativeBackend, deviceDesc);
        // The preview window is the render size and is not user-resizable, so a
        // scene whose RES is larger than the display would open a window the
        // user can neither shrink nor fully see. Scale the preview down to fit
        // (aspect preserved); headless renders keep the scene's own RES.
        auto fitPreviewSize = [&](int& w, int& h) {
            if (!preview)
                return;
            rhi::Extent2D disp = device->displaySize();
            if (disp.width <= 0 || disp.height <= 0 || (w <= disp.width && h <= disp.height))
                return;
            double scale = std::min(disp.width / (double)w, disp.height / (double)h);
            int fitW = std::max(1, (int)(w * scale));
            int fitH = std::max(1, (int)(h * scale));
            std::cout << "mini: " << w << "x" << h << " does not fit the display ("
                      << disp.width << "x" << disp.height << "), previewing at " << fitW << "x"
                      << fitH << "\n";
            w = fitW;
            h = fitH;
        };
        fitPreviewSize(width, height);

        const std::string outSuffix = std::string("_") + rhi::backendName(rhi::kNativeBackend) + ".png";
        auto stream = device->createStream();
        auto megaPipe = device->createPipeline({ "pathtraceKernel" });

        // Size-dependent session resources: the accumulation buffer, the
        // dispatch grid and (wavefront) the width*height-sized queues. Allocated
        // here and again from the scene swap when the resolution changes; the
        // old buffers free on reassignment (after the swap's waitIdle()).
        std::unique_ptr<rhi::Buffer> accum;
        size_t accumBytes = 0;
        rhi::Dim3 grid{ 1, 1, 1 };
        std::unique_ptr<rhi::Buffer> raysA, raysB, shadowQueue, qDiffuse, qConductor, qGlass;
        auto allocSizedBuffers = [&] {
            // Release the previous size's buffers before allocating the new
            // ones: assigning over them would keep both sets live at the peak,
            // and the wavefront queues are five width*height*WF_PATHSTATE_SIZE
            // allocations (gigabytes at large resolutions). Safe because every
            // caller has drained the stream first.
            accum.reset();
            raysA.reset();
            raysB.reset();
            qDiffuse.reset();
            qConductor.reset();
            qGlass.reset();
            shadowQueue.reset();
            accumBytes = (size_t)width * height * 4 * sizeof(float);
            accum = device->createBuffer({ accumBytes, rhi::MemoryLocation::Shared, "accum" });
            std::memset(accum->hostPtr(), 0, accumBytes);
            grid = rhi::Dim3{ (unsigned)(width + 15) / 16, (unsigned)(height + 15) / 16, 1 };
            if (mode == "wavefront") {
                const size_t queueBytes = (size_t)width * height * WF_PATHSTATE_SIZE;
                raysA = device->createBuffer({ queueBytes, rhi::MemoryLocation::DeviceLocal, "wf.raysA" });
                raysB = device->createBuffer({ queueBytes, rhi::MemoryLocation::DeviceLocal, "wf.raysB" });
                qDiffuse = device->createBuffer({ queueBytes, rhi::MemoryLocation::DeviceLocal, "wf.qDiffuse" });
                qConductor = device->createBuffer({ queueBytes, rhi::MemoryLocation::DeviceLocal, "wf.qConductor" });
                qGlass = device->createBuffer({ queueBytes, rhi::MemoryLocation::DeviceLocal, "wf.qGlass" });
                shadowQueue = device->createBuffer({ (size_t)width * height * WF_SHADOWRAY_SIZE,
                                                     rhi::MemoryLocation::DeviceLocal, "wf.shadow" });
            }
        };
        allocSizedBuffers();
        auto zeroAccum = [&] { std::memset(accum->hostPtr(), 0, accumBytes); };

        // First scene upload.
        SceneGpu sg = buildSceneGpu(*device, scene);
        rhi::Buffer* texHeap = &device->textureHeap();

        const rhi::Dim3 block{ 16, 16, 1 };

        // Wavefront-mode resources that do not depend on the size: queue
        // counters, indirect-args slots, and the pipelines (shade pipelines
        // specialize one kernel per material type via rhi::SpecConstant).
        std::unique_ptr<rhi::Buffer> counts, indirectArgs;
        std::unique_ptr<rhi::ComputePipeline> raygenPipe, prepIntersectPipe, prepShadePipe,
            prepShadowPipe, intersectPipe, shadeDiffusePipe, shadeConductorPipe, shadeGlassPipe,
            shadowPipe;
        if (mode == "wavefront") {
            counts = device->createBuffer({ WF_NUM_COUNTERS * 4, rhi::MemoryLocation::DeviceLocal, "wf.counts" });
            indirectArgs = device->createBuffer({ WF_NUM_ARG_SLOTS * 16, rhi::MemoryLocation::DeviceLocal, "wf.args" });
            raygenPipe = device->createPipeline({ "wf_raygen" });
            prepIntersectPipe = device->createPipeline({ "wf_prep_intersect" });
            prepShadePipe = device->createPipeline({ "wf_prep_shade" });
            prepShadowPipe = device->createPipeline({ "wf_prep_shadow" });
            intersectPipe = device->createPipeline({ "wf_intersect" });
            shadeDiffusePipe = device->createPipeline({ "wf_shade", { { 0, MINI_MAT_DIFFUSE } } });
            shadeConductorPipe = device->createPipeline({ "wf_shade", { { 0, MINI_MAT_CONDUCTOR } } });
            shadeGlassPipe = device->createPipeline({ "wf_shade", { { 0, MINI_MAT_GLASS } } });
            shadowPipe = device->createPipeline({ "wf_shadow" });
        }
        const rhi::Dim3 one{ 1, 1, 1 };
        const rhi::Dim3 tile{ PRIM_TILE, 1, 1 };

        // Renders one accumulation sample with the given params (mega single
        // kernel, or the wavefront raygen->intersect->shade bounce loop). Reads
        // the current SceneGpu by reference, so a scene swap is transparent.
        auto dispatchSample = [&](const MiniParams& p) {
            if (mode == "mega") {
                stream->dispatch(*megaPipe, grid, block, &p, sizeof(p),
                                 { accum.get(), sg.matBuf.get(), sg.objBuf.get(),
                                   sg.rt[0], sg.rt[1], sg.rt[2], texHeap,
                                   sg.normalBuf.get(), sg.uvBuf.get(),
                                   sg.spdBuf.get(), sg.r2sBuf.get(),
                                   sg.lightBuf.get(), sg.envDistBuf.get() });
                return;
            }
            stream->dispatch(*raygenPipe, grid, block, &p, sizeof(p),
                             { raysA.get(), counts.get() });
            unsigned cur = WF_COUNT_RAY_A;
            for (unsigned d = 0; d < p.maxDepth; d++) {
                unsigned next = (cur == WF_COUNT_RAY_A) ? WF_COUNT_RAY_B : WF_COUNT_RAY_A;
                rhi::Buffer* raysCur = (cur == WF_COUNT_RAY_A) ? raysA.get() : raysB.get();
                rhi::Buffer* raysNext = (cur == WF_COUNT_RAY_A) ? raysB.get() : raysA.get();

                WfCtl c = {};
                c.numObjects = p.numObjects;
                c.maxDepth = p.maxDepth;
                c.bvhNumNodes = p.bvhNumNodes;
                c.envMapIdx = p.envMapIdx;
                c.numLights = p.numLights;
                c.envW = p.envW;
                c.envH = p.envH;
                c.filmR0 = p.filmR0;
                c.filmR1 = p.filmR1;
                c.filmR2 = p.filmR2;
                c.srcCounter = cur;
                c.zeroCounter = next;

                stream->dispatch(*prepIntersectPipe, one, one, &c, sizeof(c),
                                 { counts.get(), indirectArgs.get() });
                stream->dispatchIndirect(*intersectPipe, tile, *indirectArgs,
                                         WF_ARG_INTERSECT * 16, &c, sizeof(c),
                                         { counts.get(), raysCur, sg.objBuf.get(),
                                           sg.rt[0], sg.rt[1], sg.rt[2], sg.matBuf.get(), accum.get(),
                                           qDiffuse.get(), qConductor.get(), qGlass.get(),
                                           texHeap, sg.normalBuf.get(), sg.uvBuf.get(),
                                           sg.spdBuf.get(), sg.r2sBuf.get(), sg.envDistBuf.get() });
                stream->dispatch(*prepShadePipe, one, one, &c, sizeof(c),
                                 { counts.get(), indirectArgs.get() });

                c.dstCounter = next;
                struct { rhi::ComputePipeline* pipe; rhi::Buffer* queue; unsigned counter; unsigned argSlot; }
                shadePasses[] = {
                    { shadeDiffusePipe.get(), qDiffuse.get(), WF_COUNT_SHADE_DIFFUSE, WF_ARG_DIFFUSE },
                    { shadeConductorPipe.get(), qConductor.get(), WF_COUNT_SHADE_CONDUCTOR, WF_ARG_CONDUCTOR },
                    { shadeGlassPipe.get(), qGlass.get(), WF_COUNT_SHADE_GLASS, WF_ARG_GLASS },
                };
                for (const auto& pass : shadePasses) {
                    c.srcCounter = pass.counter;
                    stream->dispatchIndirect(*pass.pipe, tile, *indirectArgs,
                                             pass.argSlot * 16, &c, sizeof(c),
                                             { counts.get(), pass.queue, raysNext,
                                               sg.matBuf.get(), texHeap,
                                               sg.spdBuf.get(), sg.r2sBuf.get(),
                                               sg.lightBuf.get(), sg.objBuf.get(),
                                               sg.rt[1], sg.rt[2], sg.envDistBuf.get(),
                                               shadowQueue.get() });
                }
                // Next-event estimation shadow rays for this bounce, before the
                // next intersect so per-pixel contributions land in the same
                // order as the megakernel's.
                stream->dispatch(*prepShadowPipe, one, one, &c, sizeof(c),
                                 { counts.get(), indirectArgs.get() });
                stream->dispatchIndirect(*shadowPipe, tile, *indirectArgs,
                                         WF_ARG_SHADOW * 16, &c, sizeof(c),
                                         { counts.get(), shadowQueue.get(), sg.objBuf.get(),
                                           sg.rt[0], sg.rt[1], sg.rt[2], accum.get() });
                cur = next;
            }
        };

        const glm::mat3 filmM = srgbRgbFromXyz();

        // Base params: film + scene fields. Camera is filled per branch below.
        // One writer for every SceneGpu-derived param so the initial setup and
        // scene hot-swap can't drift apart field-by-field.
        MiniParams params = {};
        auto applySceneParams = [&] {
            setFilmParams(params, filmM, width, height, sg.fovyDeg);
            params.maxDepth = sg.maxDepth;
            params.numObjects = sg.numObjects;
            params.bvhNumNodes = sg.bvhNumNodes;
            params.envMapIdx = sg.envMapIdx;
            params.numLights = sg.numLights;
            params.envW = sg.envW;
            params.envH = sg.envH;
        };
        applySceneParams();

        // ------------------------------------------------------------------
        // Headless: the original accumulate-N-samples loop, camera built with
        // the exact scene.cpp basis so PNGs stay bitwise-identical.
        // ------------------------------------------------------------------
        if (!preview) {
            setCameraExact(params, sg.eye, sg.lookAt, sg.up);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < spp; i++) {
                params.iter = (unsigned)i;
                dispatchSample(params);
                if ((i + 1) % 4 == 0)
                    stream->submit();
                if (spp >= 10 && (i + 1) % (spp / 10) == 0)
                    std::cout << "  " << (i + 1) << "/" << spp << " spp\r" << std::flush;
            }
            stream->waitIdle();
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cout << "\nmini: rendered " << spp << " spp in " << sec << " s ("
                      << (sec > 0 ? spp / sec : 0.0) << " spp/s, " << mode << ", "
                      << rhi::backendName(rhi::kNativeBackend) << ")\n";
            // MINI_DUMP_ACCUM=<path>: also write the raw float accumulator
            // (width*height RGBA32F, radiance sums before /spp and tonemap).
            // Float-precision output for FurnaceTest and parity debugging —
            // the 8-bit PNG quantizes away sub-0.4% differences.
            if (const char* dumpPath = std::getenv("MINI_DUMP_ACCUM")) {
                std::ofstream f(dumpPath, std::ios::binary);
                f.write((const char*)accum->hostPtr(), (std::streamsize)accumBytes);
                if (!f) {
                    std::cerr << "failed to write " << dumpPath << "\n";
                    return 1;
                }
            }
            std::string outName = !outOverride.empty() ? outOverride : sg.outputName + outSuffix;
            if (!savePng(outName, (const float*)accum->hostPtr(), width, height, spp)) {
                std::cerr << "failed to write " << outName << "\n";
                return 1;
            }
            std::cout << "wrote " << outName << "\n";
            return 0;
        }

        // ------------------------------------------------------------------
        // Interactive preview: fly camera + ImGui overlay + scene hot-swap.
        // ------------------------------------------------------------------
        gui::FlyCamera cam = gui::FlyCamera::fromLookAt(sg.eye, sg.lookAt);
        setCameraFly(params, cam);

        gui::State ui;
        ui.camera = &cam;
        ui.sceneNames = sceneNames;
        ui.selectedScene = selectedScene;
        ui.stats.targetSpp = spp;
        ui.stats.mode = mode;
        auto applySceneStats = [&] {
            ui.stats.numObjects = (int)sg.numObjects;
            ui.stats.numTris = sg.numTris;
            ui.stats.maxDepth = sg.maxDepth;
        };
        applySceneStats();
        device->enableGui([&] { gui::draw(ui); });

        rhi::Buffer* presentBuf = &device->presentTarget(width, height);
        auto tonemapPipe = device->createPipeline({ "present_tonemap" });

        auto outNameFor = [&] {
            return !outOverride.empty() ? outOverride : sg.outputName + outSuffix;
        };

        // Swap in a different scene: drain the GPU (old buffers are about to be
        // freed), rebuild resources, follow the new scene's resolution (window,
        // accumulation, queues), reset the camera. Costs a BVH build + reupload
        // (dropdown-speed, not live); the loop zeroes accumulation afterwards.
        // Uploads a loaded scene and follows its resolution. Requires a drained
        // stream (it frees the previous scene's buffers) and throws on a failed
        // GPU allocation, which the caller recovers from.
        auto activateScene = [&](const CoreScene& s) {
            // Free the old scene before building the new one so its bindless
            // texture slots return to the heap freelist and the new upload
            // reuses them, instead of both scenes' textures being live at once
            // (which would push the slot high-water toward the 1024 cap on
            // repeated swaps). Safe after waitIdle(): no dispatch still reads it.
            sg = {};
            sg = buildSceneGpu(*device, s);
            int w = s.camera.width, h = s.camera.height;
            fitPreviewSize(w, h);
            if (w != width || h != height) {
                width = w;
                height = h;
                allocSizedBuffers();
                presentBuf = &device->presentTarget(width, height);
                std::cout << "mini: window resized to " << width << "x" << height << "\n";
            }
        };

        auto swapScene = [&](int idx) {
            if (idx < 0 || idx >= (int)scenePaths.size())
                return;
            CoreScene next;
            std::string e;
            if (!loadScene(scenePaths[idx], next, e)) {
                std::cout << "mini: failed to load " << scenePaths[idx] << ": " << e << "\n";
                return;
            }
            stream->waitIdle();
            int shown = idx;
            try {
                activateScene(next);
            } catch (const std::exception& ex) {
                // Typically an out-of-memory building a much larger scene's
                // buffers. The scene we came from is already released by now, so
                // rebuild it rather than letting the throw end the session and
                // discard the accumulated image; only a failure to restore
                // (nothing left to render) is fatal.
                std::cout << "mini: failed to switch to " << sceneNames[idx] << ": " << ex.what()
                          << "\n";
                shown = ui.selectedScene;
                CoreScene back;
                std::string backErr;
                if (shown == idx || !loadScene(scenePaths[shown], back, backErr))
                    throw;
                activateScene(back);
                std::cout << "mini: restored " << sceneNames[shown] << "\n";
            }
            applySceneParams();
            cam = gui::FlyCamera::fromLookAt(sg.eye, sg.lookAt);
            setCameraFly(params, cam);
            ui.selectedScene = shown;
            applySceneStats();
            if (shown == idx)
                std::cout << "mini: switched to " << sceneNames[idx] << "\n";
        };

        // Renderer-side hooks for the portable preview loop (gui::runPreview owns
        // sample accounting, ~60 Hz pacing, and accumulation restart).
        gui::PreviewHooks hooks;
        hooks.renderSample = [&](int iter) {
            params.iter = (unsigned)iter;
            dispatchSample(params);
            stream->submit();
        };
        hooks.presentFrame = [&](int samples) {
            MiniParams tp = params;
            tp.iter = (unsigned)samples;
            stream->dispatch(*tonemapPipe, grid, block, &tp, sizeof(tp),
                             { accum.get(), presentBuf });
            stream->submit();
            return device->present();
        };
        hooks.applyCamera = [&] { setCameraFly(params, cam); };
        // PreviewHooks contract: anything touching accum host-side drains the
        // stream first — renderSample leaves up to kMaxInFlight command buffers
        // still writing accum, so a bare memset/read here would race the GPU.
        hooks.zeroAccum = [&] {
            stream->waitIdle();
            zeroAccum();
        };
        hooks.loadScene = swapScene;
        int previewExit = 0;
        auto saveNow = [&](int samples) {
            stream->waitIdle();
            std::string outName = outNameFor();
            if (savePng(outName, (const float*)accum->hostPtr(), width, height, samples)) {
                std::cout << "wrote " << outName << " (" << samples << " spp)\n";
                return true;
            }
            std::cerr << "failed to write " << outName << "\n";
            return false;
        };
        hooks.save = [&](int samples) { saveNow(samples); };
        hooks.finish = [&](int samples) {
            if (!saveNow(samples))
                previewExit = 1;
        };

        gui::runPreview(ui, spp, hooks);
        return previewExit;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
