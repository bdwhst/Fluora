// FluoraMini: Cornell-box vertical slice on the Metal RHI backend (M1 in
// docs/metal-rhi-design.md). Loads a Fluora .txt scene subset, path-traces it
// via rhi:: with a live preview window (one frame per iteration, frozen at the
// last frame until closed), and writes a PNG.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <stb_image_write.h>

#include <glm/glm.hpp>

#include "../core/host_math.h"
#include "../core/image_loader.h"
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

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "usage: FluoraMini <scene.txt> [--spp N] [--out name.png]"
                     " [--mode wavefront|mega] [--no-preview]\n";
        return 1;
    }
    std::string scenePath = argv[1];
    int sppOverride = -1;
    std::string outOverride;
    std::string mode = "wavefront";
    bool preview = true;
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--spp" && i + 1 < argc)
            sppOverride = std::atoi(argv[++i]);
        else if (arg == "--out" && i + 1 < argc)
            outOverride = argv[++i];
        else if (arg == "--mode" && i + 1 < argc)
            mode = argv[++i];
        else if (arg == "--no-preview")
            preview = false;
    }
    if (mode != "wavefront" && mode != "mega") {
        std::cerr << "unknown --mode " << mode << "\n";
        return 1;
    }

    CoreScene scene;
    std::string err;
    if (!loadTxtScene(scenePath, scene, err)) {
        std::cerr << "scene load failed: " << err << "\n";
        return 1;
    }
    const int width = scene.camera.width;
    const int height = scene.camera.height;
    const int spp = sppOverride > 0 ? sppOverride : scene.camera.iterations;
    std::cout << "mini: " << scene.objects.size() << " objects, " << scene.materials.size()
              << " materials, " << width << "x" << height << ", " << spp << " spp, "
              << mode << " mode\n";

    // CoreScene -> device PODs, resolving named spectra to dense-table
    // offsets (the CoreMaterial carries the names; SpectralTables densifies
    // on demand). This mapping is the mini scaffolding now; it dissolves as
    // real materials port to src/core.
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
    std::vector<MiniMaterial> materials;
    materials.reserve(scene.materials.size());
    for (const auto& m : scene.materials) {
        MiniMaterial mm = {};
        switch (m.type) {
        case CoreMaterialType::Diffuse: mm.type = MINI_MAT_DIFFUSE; break;
        case CoreMaterialType::Emissive: mm.type = MINI_MAT_EMITTING; break;
        case CoreMaterialType::Dielectric: mm.type = MINI_MAT_GLASS; break;
        case CoreMaterialType::Conductor: mm.type = MINI_MAT_CONDUCTOR; break;
        }
        mm.rgb = hostStore3(m.rgb);
        mm.emittance = m.emittance;
        mm.ior = m.ior;
        mm.roughness = m.roughness;
        mm.texIdx = m.texIdx == kCoreTexNone ? MINI_TEX_NONE : m.texIdx;
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

    try {
        // Runtime MSL compile: shared structs + RHI primitives (the wavefront
        // kernels use prim_queue_alloc) + renderer kernels, concatenated
        // (see DeviceDesc::shaderSource in rhi.h).
        rhi::DeviceDesc deviceDesc;
        deviceDesc.shaderSource = readTextFile(std::string(RHI_SHADER_DIR) + "/gpu_portable.h")
                                + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/spectrum_shared.h")
                                + "\n" + readTextFile(std::string(MINI_SHADER_DIR) + "/mini_shared.h")
                                + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/accel_shared.h")
                                + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/bsdf_shared.h")
                                + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/tonemap_shared.h")
                                + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/envmap_shared.h")
                                + "\n" + readTextFile(std::string(RHI_SHADER_DIR) + "/primitives_shared.h")
                                + "\n" + readTextFile(std::string(RHI_SHADER_DIR) + "/primitives_gpu.h")
                                + "\n" + readTextFile(std::string(RHI_SHADER_DIR) + "/raytrace_gpu.h")
                                + "\n" + readTextFile(std::string(RHI_SHADER_DIR) + "/texture.metal")
                                + "\n" + readTextFile(std::string(MINI_SHADER_DIR) + "/pathtrace_gpu.h");
        auto device = rhi::createDevice(rhi::BackendKind::Metal, deviceDesc);
        auto stream = device->createStream();
        auto pipeline = device->createPipeline({ "pathtraceKernel" });

        const size_t accumBytes = (size_t)width * height * 4 * sizeof(float);
        auto accum = device->createBuffer({ accumBytes, rhi::MemoryLocation::Shared, "accum" });
        std::memset(accum->hostPtr(), 0, accumBytes);

        auto matBuf = device->createBuffer(
            { std::max<size_t>(materials.size(), 1) * sizeof(MiniMaterial),
              rhi::MemoryLocation::Shared, "materials" });
        std::memcpy(matBuf->hostPtr(), materials.data(),
                    materials.size() * sizeof(MiniMaterial));
        auto objBuf = device->createBuffer(
            { std::max<size_t>(objects.size(), 1) * sizeof(MiniObject),
              rhi::MemoryLocation::Shared, "objects" });
        std::memcpy(objBuf->hostPtr(), objects.data(),
                    objects.size() * sizeof(MiniObject));

        // Mesh geometry lives behind the ray-tracing seam: the CPU-built BVH
        // (core/bvh_builder) is handed to the RayIntersector, whose buffers we
        // slot-bind for rt_closest_hit.
        auto intersector = device->createIntersector();
        rhi::AccelBuildInput accel;
        accel.nodes = scene.bvh.nodes.data();
        accel.nodeBytes = scene.bvh.nodes.size() * sizeof(RtBvhNode);
        accel.numNodesPerDir = scene.bvh.numNodesPerDir;
        accel.triangles = scene.tris.data();
        accel.triangleBytes = scene.tris.size() * sizeof(gpu_uint4);
        accel.positions = scene.positions.data();
        accel.positionBytes = scene.positions.size() * sizeof(gpu_storage3);
        intersector->build(accel);
        auto rt = intersector->bindings();  // {nodes, tris, positions}

        // Mesh vertex attributes (unified indices with positions); dummy-sized
        // when the scene has no meshes (Metal rejects zero-length buffers).
        auto makeUpload = [&](const void* data, size_t bytes, const char* name) {
            auto buf = device->createBuffer(
                { std::max<size_t>(bytes, 16), rhi::MemoryLocation::Shared, name });
            if (data && bytes)
                std::memcpy(buf->hostPtr(), data, bytes);
            return buf;
        };
        auto normalBuf = makeUpload(scene.normals.data(),
                                    scene.normals.size() * sizeof(gpu_storage3), "normals");
        auto uvBuf = makeUpload(scene.uvs.data(),
                                scene.uvs.size() * sizeof(gpu_float2), "uvs");

        // Base-color textures first, in texturePaths order, so heap indices
        // match MiniMaterial.texIdx (invariant I-1: index, not handle). A 1x1
        // white fallback keeps indices aligned when a file fails to load.
        std::vector<std::unique_ptr<rhi::Texture>> matTextures;
        for (const auto& texPath : scene.texturePaths) {
            LdrImage img;
            if (!loadLdrImage(texPath, img)) {
                std::cout << "mini: failed to load texture " << texPath << ", using white\n";
                img.width = img.height = 1;
                img.rgba = { 255, 255, 255, 255 };
            }
            auto tex = device->createTexture(
                { img.width, img.height, rhi::TextureFormat::RGBA8Unorm,
                  true, /*srgb=*/true, "basecolor" });
            tex->upload(img.rgba.data(), img.rgba.size());
            matTextures.push_back(std::move(tex));
        }
        if (!scene.texturePaths.empty())
            std::cout << "mini: " << scene.texturePaths.size() << " material textures\n";

        // Environment map: an RGBA32F texture in the bindless heap; kernels
        // get the heap index through params (invariant I-1: index, not handle).
        uint32_t envMapIdx = MINI_ENV_NONE;
        std::unique_ptr<rhi::Texture> envTex;
        if (!scene.envMapPath.empty()) {
            HdrImage envImg;
            if (loadHdrImage(scene.envMapPath, envImg)) {
                envTex = device->createTexture(
                    { envImg.width, envImg.height, rhi::TextureFormat::RGBA32Float,
                      true, false, "envmap" });
                envTex->upload(envImg.rgba.data(), envImg.rgba.size() * sizeof(float));
                envMapIdx = (uint32_t)envTex->shaderHandle();
                std::cout << "mini: environment map " << scene.envMapPath << " ("
                          << envImg.width << "x" << envImg.height << ")\n";
            } else {
                std::cout << "mini: failed to load environment map "
                          << scene.envMapPath << ", using black\n";
            }
        }
        rhi::Buffer* texHeap = &device->textureHeap();

        // Spectral tables (invariant I-1: kernels get offsets, not pointers):
        // the dense-spectra buffer (CIE curves, D65, named eta/k) and the sRGB
        // rgb2spec coefficient table (zNodes then coeffs).
        auto spdBuf = device->createBuffer(
            { spectra.buffer().size() * sizeof(float), rhi::MemoryLocation::Shared, "spd" });
        std::memcpy(spdBuf->hostPtr(), spectra.buffer().data(),
                    spectra.buffer().size() * sizeof(float));
        Rgb2SpecView r2sView = rgb2specSrgb();
        auto r2sBuf = device->createBuffer(
            { (r2sView.zNodeCount + r2sView.coeffCount) * sizeof(float),
              rhi::MemoryLocation::Shared, "rgb2spec" });
        std::memcpy(r2sBuf->hostPtr(), r2sView.zNodes, r2sView.zNodeCount * sizeof(float));
        std::memcpy((float*)r2sBuf->hostPtr() + r2sView.zNodeCount, r2sView.coeffs,
                    r2sView.coeffCount * sizeof(float));

        glm::mat3 filmM = srgbRgbFromXyz();

        // Camera setup replicating scene.cpp exactly, quirks included: pixelLength
        // uses tan(fovy in degrees->radians) un-halved, and the UP vector is used
        // as given rather than re-orthogonalized.
        MiniParams params = {};
        glm::vec3 camView = glm::normalize(scene.camera.lookAt - scene.camera.eye);
        params.camPos = hostStore3(scene.camera.eye);
        params.camView = hostStore3(camView);
        params.camUp = hostStore3(scene.camera.up);
        params.camRight = hostStore3(glm::normalize(glm::cross(camView, scene.camera.up)));
        // Row i of the film matrix = (M[0][i], M[1][i], M[2][i]) (glm is
        // column-major); kernels apply rgb = (dot(r0,xyz), ...).
        params.filmR0 = hostStore3(glm::vec3(filmM[0][0], filmM[1][0], filmM[2][0]));
        params.filmR1 = hostStore3(glm::vec3(filmM[0][1], filmM[1][1], filmM[2][1]));
        params.filmR2 = hostStore3(glm::vec3(filmM[0][2], filmM[1][2], filmM[2][2]));
        float yscaled = std::tan(scene.camera.fovyDeg * 3.14159265358979323846f / 180.0f);
        float xscaled = yscaled * width / (float)height;
        params.pixelLenX = 2.0f * xscaled / (float)width;
        params.pixelLenY = 2.0f * yscaled / (float)height;
        params.width = width;
        params.height = height;
        params.maxDepth = scene.camera.maxDepth;
        params.numObjects = (unsigned)objects.size();
        params.bvhNumNodes = intersector->numNodes();
        params.envMapIdx = envMapIdx;

        const rhi::Dim3 grid{ (unsigned)(width + 15) / 16, (unsigned)(height + 15) / 16, 1 };
        const rhi::Dim3 block{ 16, 16, 1 };

        // Wavefront-mode resources: ping-pong ray queues + one shade queue per
        // material type (GPU only, host never reads paths), queue counters,
        // indirect-args slots. Shade pipelines are one kernel specialized per
        // material type via rhi::SpecConstant.
        std::unique_ptr<rhi::Buffer> raysA, raysB, counts, indirectArgs;
        std::unique_ptr<rhi::Buffer> qDiffuse, qConductor, qGlass;
        std::unique_ptr<rhi::ComputePipeline> raygenPipe, prepIntersectPipe, prepShadePipe,
            intersectPipe, shadeDiffusePipe, shadeConductorPipe, shadeGlassPipe;
        if (mode == "wavefront") {
            const size_t queueBytes = (size_t)width * height * WF_PATHSTATE_SIZE;
            raysA = device->createBuffer({ queueBytes, rhi::MemoryLocation::DeviceLocal, "wf.raysA" });
            raysB = device->createBuffer({ queueBytes, rhi::MemoryLocation::DeviceLocal, "wf.raysB" });
            qDiffuse = device->createBuffer({ queueBytes, rhi::MemoryLocation::DeviceLocal, "wf.qDiffuse" });
            qConductor = device->createBuffer({ queueBytes, rhi::MemoryLocation::DeviceLocal, "wf.qConductor" });
            qGlass = device->createBuffer({ queueBytes, rhi::MemoryLocation::DeviceLocal, "wf.qGlass" });
            counts = device->createBuffer({ WF_NUM_COUNTERS * 4, rhi::MemoryLocation::DeviceLocal, "wf.counts" });
            indirectArgs = device->createBuffer({ WF_NUM_ARG_SLOTS * 16, rhi::MemoryLocation::DeviceLocal, "wf.args" });
            raygenPipe = device->createPipeline({ "wf_raygen" });
            prepIntersectPipe = device->createPipeline({ "wf_prep_intersect" });
            prepShadePipe = device->createPipeline({ "wf_prep_shade" });
            intersectPipe = device->createPipeline({ "wf_intersect" });
            shadeDiffusePipe = device->createPipeline({ "wf_shade", { { 0, MINI_MAT_DIFFUSE } } });
            shadeConductorPipe = device->createPipeline({ "wf_shade", { { 0, MINI_MAT_CONDUCTOR } } });
            shadeGlassPipe = device->createPipeline({ "wf_shade", { { 0, MINI_MAT_GLASS } } });
        }
        const rhi::Dim3 one{ 1, 1, 1 };
        const rhi::Dim3 tile{ PRIM_TILE, 1, 1 };

        // Preview: a tonemap dispatch writes the running average into the RHI
        // present target, then present() blits it to the window. Rate-limited
        // to ~60 Hz so fast scenes are not throttled by the drawable pool;
        // slow scenes present every iteration.
        rhi::Buffer* presentBuf = nullptr;
        std::unique_ptr<rhi::ComputePipeline> tonemapPipe;
        if (preview) {
            presentBuf = &device->presentTarget(width, height);
            tonemapPipe = device->createPipeline({ "present_tonemap" });
        }
        using clock = std::chrono::steady_clock;
        auto lastPresent = clock::now() - std::chrono::seconds(1);
        bool closed = false;
        int completed = 0;

        for (int i = 0; i < spp; i++) {
            params.iter = (unsigned)i;
            if (mode == "mega") {
                stream->dispatch(*pipeline, grid, block, &params, sizeof(params),
                                 { accum.get(), matBuf.get(), objBuf.get(),
                                   rt[0], rt[1], rt[2], texHeap,
                                   normalBuf.get(), uvBuf.get(),
                                   spdBuf.get(), r2sBuf.get() });
            } else {
                stream->dispatch(*raygenPipe, grid, block, &params, sizeof(params),
                                 { raysA.get(), counts.get() });
                unsigned cur = WF_COUNT_RAY_A;
                for (unsigned d = 0; d < params.maxDepth; d++) {
                    unsigned next = (cur == WF_COUNT_RAY_A) ? WF_COUNT_RAY_B : WF_COUNT_RAY_A;
                    rhi::Buffer* raysCur = (cur == WF_COUNT_RAY_A) ? raysA.get() : raysB.get();
                    rhi::Buffer* raysNext = (cur == WF_COUNT_RAY_A) ? raysB.get() : raysA.get();

                    WfCtl c = {};
                    c.numObjects = params.numObjects;
                    c.maxDepth = params.maxDepth;
                    c.bvhNumNodes = params.bvhNumNodes;
                    c.envMapIdx = envMapIdx;
                    c.filmR0 = params.filmR0;
                    c.filmR1 = params.filmR1;
                    c.filmR2 = params.filmR2;
                    c.srcCounter = cur;
                    c.zeroCounter = next;

                    stream->dispatch(*prepIntersectPipe, one, one, &c, sizeof(c),
                                     { counts.get(), indirectArgs.get() });
                    stream->dispatchIndirect(*intersectPipe, tile, *indirectArgs,
                                             WF_ARG_INTERSECT * 16, &c, sizeof(c),
                                             { counts.get(), raysCur, objBuf.get(),
                                               rt[0], rt[1], rt[2], matBuf.get(), accum.get(),
                                               qDiffuse.get(), qConductor.get(), qGlass.get(),
                                               texHeap, normalBuf.get(), uvBuf.get(),
                                               spdBuf.get(), r2sBuf.get() });
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
                                                   matBuf.get(), texHeap,
                                                   spdBuf.get(), r2sBuf.get() });
                    }
                    cur = next;
                }
            }
            completed = i + 1;
            if (preview) {
                // Per-sample submit: the stream's bounded in-flight ring paces
                // the CPU to GPU progress, so the wall clock below measures
                // actual render progress rather than encode speed.
                stream->submit();
                auto now = clock::now();
                if (completed == spp
                    || now - lastPresent >= std::chrono::milliseconds(16)) {
                    MiniParams tp = params;
                    tp.iter = (unsigned)completed;
                    stream->dispatch(*tonemapPipe, grid, block, &tp, sizeof(tp),
                                     { accum.get(), presentBuf });
                    stream->submit();
                    lastPresent = now;
                    if (!device->present()) {
                        closed = true;
                        break;
                    }
                }
            } else if ((i + 1) % 4 == 0) {
                stream->submit();
            }
            if (spp >= 10 && (i + 1) % (spp / 10) == 0)
                std::cout << "  " << (i + 1) << "/" << spp << " spp\r" << std::flush;
        }
        stream->waitIdle();
        std::cout << "\n";
        if (closed)
            std::cout << "preview closed at " << completed << "/" << spp
                      << " spp, saving partial image\n";

        const float* acc = (const float*)accum->hostPtr();
        std::vector<unsigned char> pixels((size_t)width * height * 3);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // saveImage() in main.cpp writes setPixel(width-1-x, y): saved
                // renders are mirrored relative to kernel indexing. Match it so
                // images are comparable with img/ references.
                size_t src = (size_t)y * width + x;
                size_t dst = (size_t)y * width + (width - 1 - x);
                gpu_float3 v = tonemap_aces(gpu_float3(
                    acc[src * 4 + 0], acc[src * 4 + 1], acc[src * 4 + 2])
                    / (float)completed);
                for (int ch = 0; ch < 3; ch++)
                    pixels[dst * 3 + ch] = (unsigned char)(v[ch] * 255.0f + 0.5f);
            }
        }
        std::string outName = !outOverride.empty() ? outOverride
                                                   : scene.camera.outputName + "_metal.png";
        if (!stbi_write_png(outName.c_str(), width, height, 3, pixels.data(), width * 3)) {
            std::cerr << "failed to write " << outName << "\n";
            return 1;
        }
        std::cout << "wrote " << outName << "\n";

        // Freeze at the last frame: keep the window alive (present() re-blits
        // the final image and pumps events) until the user closes it.
        if (preview && !closed) {
            std::cout << "preview: close the window (or press q / Esc) to exit\n";
            while (device->present())
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
