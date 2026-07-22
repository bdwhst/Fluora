// FluoraMini: Cornell-box vertical slice on the Metal RHI backend (M1 in
// docs/metal-rhi-design.md). Loads a Fluora .txt scene subset, path-traces it
// with a megakernel via rhi::, tonemaps on the CPU with the same ACES curve as
// the CUDA preview, and writes a PNG.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <stb_image_write.h>

#include "../rhi/rhi.h"
#include "mini_scene.h"

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

// Same curve as util_postprocess_ACESFilm / sendImageToPBO (no extra gamma,
// matching the CUDA preview output).
float acesFilm(float x)
{
    float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    return std::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "usage: FluoraMini <scene.txt> [--spp N] [--out name.png]\n";
        return 1;
    }
    std::string scenePath = argv[1];
    int sppOverride = -1;
    std::string outOverride;
    for (int i = 2; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--spp")
            sppOverride = std::atoi(argv[i + 1]);
        else if (std::string(argv[i]) == "--out")
            outOverride = argv[i + 1];
    }

    MiniScene scene;
    std::string err;
    if (!miniLoadScene(scenePath, scene, err)) {
        std::cerr << "scene load failed: " << err << "\n";
        return 1;
    }
    const int width = scene.camera.width;
    const int height = scene.camera.height;
    const int spp = sppOverride > 0 ? sppOverride : scene.camera.iterations;
    std::cout << "mini: " << scene.objects.size() << " objects, " << scene.materials.size()
              << " materials, " << width << "x" << height << ", " << spp << " spp\n";

    try {
        // Runtime MSL compile: shared structs header + kernel source, concatenated
        // (see DeviceDesc::shaderSource in rhi.h).
        rhi::DeviceDesc deviceDesc;
        deviceDesc.shaderSource = readTextFile(std::string(MINI_SHADER_DIR) + "/mini_shared.h")
                                + "\n" + readTextFile(std::string(MINI_SHADER_DIR) + "/pathtrace.metal");
        auto device = rhi::createDevice(rhi::BackendKind::Metal, deviceDesc);
        auto stream = device->createStream();
        auto pipeline = device->createPipeline({ "pathtraceKernel" });

        const size_t accumBytes = (size_t)width * height * 4 * sizeof(float);
        auto accum = device->createBuffer({ accumBytes, rhi::MemoryLocation::Shared, "accum" });
        std::memset(accum->hostPtr(), 0, accumBytes);

        auto matBuf = device->createBuffer(
            { std::max<size_t>(scene.materials.size(), 1) * sizeof(MiniMaterial),
              rhi::MemoryLocation::Shared, "materials" });
        std::memcpy(matBuf->hostPtr(), scene.materials.data(),
                    scene.materials.size() * sizeof(MiniMaterial));
        auto objBuf = device->createBuffer(
            { scene.objects.size() * sizeof(MiniObject), rhi::MemoryLocation::Shared, "objects" });
        std::memcpy(objBuf->hostPtr(), scene.objects.data(), objBuf->size());

        // Camera setup replicating scene.cpp exactly, quirks included: pixelLength
        // uses tan(fovy in degrees->radians) un-halved, and the UP vector is used
        // as given rather than re-orthogonalized.
        MiniParams params = {};
        params.camPos = scene.camera.eye;
        params.camView = simd_normalize(scene.camera.lookAt - scene.camera.eye);
        params.camUp = scene.camera.up;
        params.camRight = simd_normalize(simd_cross(params.camView, scene.camera.up));
        float yscaled = std::tan(scene.camera.fovyDeg * 3.14159265358979323846f / 180.0f);
        float xscaled = yscaled * width / (float)height;
        params.pixelLenX = 2.0f * xscaled / (float)width;
        params.pixelLenY = 2.0f * yscaled / (float)height;
        params.width = width;
        params.height = height;
        params.maxDepth = scene.camera.maxDepth;
        params.numObjects = (unsigned)scene.objects.size();

        const rhi::Dim3 grid{ (unsigned)(width + 15) / 16, (unsigned)(height + 15) / 16, 1 };
        const rhi::Dim3 block{ 16, 16, 1 };
        for (int i = 0; i < spp; i++) {
            params.iter = (unsigned)i;
            stream->dispatch(*pipeline, grid, block, &params, sizeof(params),
                             { accum.get(), matBuf.get(), objBuf.get() });
            if ((i + 1) % 8 == 0)
                stream->submit();
            if (spp >= 10 && (i + 1) % (spp / 10) == 0)
                std::cout << "  " << (i + 1) << "/" << spp << " spp\r" << std::flush;
        }
        stream->waitIdle();
        std::cout << "\n";

        const float* acc = (const float*)accum->hostPtr();
        std::vector<unsigned char> pixels((size_t)width * height * 3);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // saveImage() in main.cpp writes setPixel(width-1-x, y): saved
                // renders are mirrored relative to kernel indexing. Match it so
                // images are comparable with img/ references.
                size_t src = (size_t)y * width + x;
                size_t dst = (size_t)y * width + (width - 1 - x);
                for (int ch = 0; ch < 3; ch++) {
                    float v = acesFilm(acc[src * 4 + ch] / (float)spp);
                    pixels[dst * 3 + ch] = (unsigned char)(v * 255.0f + 0.5f);
                }
            }
        }
        std::string outName = !outOverride.empty() ? outOverride
                                                   : scene.camera.outputName + "_metal.png";
        if (!stbi_write_png(outName.c_str(), width, height, 3, pixels.data(), width * 3)) {
            std::cerr << "failed to write " << outName << "\n";
            return 1;
        }
        std::cout << "wrote " << outName << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
