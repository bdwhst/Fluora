// SharedHostTest (docs/portable-device-code.md §3): proves the shared device
// code compiles in the non-Metal personality's spelling (this file compiles it
// as host C++ — the same personality nvcc sees modulo qualifiers) AND agrees
// numerically with the MSL personality: the same headers plus the same probe
// body (shared_probe.h) are compiled by the Metal compiler at runtime and
// evaluated on the GPU. Tolerance-based compare (Metal builds with fast-math).
// Exits nonzero on failure.
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "../rhi/gpu_portable.h"
#include "../core/spectrum_shared.h"
#include "../core/bsdf_shared.h"
#include "../core/envmap_shared.h"
#include "../core/light_shared.h"
#include "../core/tonemap_shared.h"
#include "shared_probe.h"

#include <glm/gtc/matrix_transform.hpp>

#include "../core/host_math.h"
#include "../core/lights.h"
#include "../core/spectra.h"
#include "../rhi/rhi.h"
#include "test_util.h"

namespace {

constexpr int kSlots = PROBE_SLOT_COUNT;

} // namespace

int main()
{
    // Spectral tables, shared by both personalities: the host pass reads the
    // vectors directly, the GPU pass reads the same bytes uploaded to buffers.
    SpectralTables tables;
    (void)tables.namedOffset("glass-Fake");   // exercise a named spectrum too
    Rgb2SpecView r2sView = rgb2specSrgb();
    std::vector<float> r2sHost(r2sView.zNodeCount + r2sView.coeffCount);
    std::memcpy(r2sHost.data(), r2sView.zNodes, r2sView.zNodeCount * sizeof(float));
    std::memcpy(r2sHost.data() + r2sView.zNodeCount, r2sView.coeffs,
                r2sView.coeffCount * sizeof(float));

    // Light-sampling fixtures. Two object-to-world transforms built in glm
    // and stored through host_math.h::hostStore4x4 — the exact same bytes
    // feed both personalities, so the GPU pass compares identical floats —
    // and a 4x2 env map run through the real host CDF builder
    // (core/lights.cpp), with one texel clamped by maxRadiance and one at
    // zero luminance.
    const gpu_storage4x4 xfHost[2] = {
        // sphere: uniform scale 3 + translate
        hostStore4x4(glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, -0.5f))
                     * glm::scale(glm::mat4(1.0f), glm::vec3(3.0f))),
        // cube: rotate 30 deg about Y * scale(2, 1, 0.5) + translate
        hostStore4x4(glm::translate(glm::mat4(1.0f), glm::vec3(-0.3f, 0.8f, 0.2f))
                     * glm::rotate(glm::mat4(1.0f), glm::radians(30.0f),
                                   glm::vec3(0.0f, 1.0f, 0.0f))
                     * glm::scale(glm::mat4(1.0f), glm::vec3(2.0f, 1.0f, 0.5f))),
    };
    HdrImage envImg;
    envImg.width = 4;
    envImg.height = 2;
    envImg.rgba = {
        1, 0, 0, 1,    0, 2, 0, 1,            0, 0, 3, 1,   4, 4, 4, 1,
        50, 0, 0, 1,   0.5f, 0.5f, 0.5f, 1,   0, 0, 0, 1,   1, 1, 1, 1,
    };
    std::vector<float> envDist = buildEnvDistribution(envImg, glm::vec3(10.0f));

    // Host personality evaluation.
    gpu_float4 host[kSlots] = {};
    PROBE_BODY(host, r2sHost.data(), tables.buffer().data(), xfHost,
               envDist.data())

    // Sample <-> pdf identities of the light slots (checked host-side once;
    // the parity compare below then carries them to the GPU personality):
    // the solid-angle pdf of the tri sample must match light_pdf_area of the
    // same point, envdist_sample's pdf must match envdist_pdf at the sampled
    // uv, env_sample_dir's pdf must match env_pdf_dir of the sampled wi, the
    // equirect uv->dir->uv round trip must return, and back-side sampling
    // must yield exact zeros (one-sided lights).
    int identityFails = 0;
    auto ident = [&](float a, float b, const char* name) {
        if (std::fabs(a - b) > 1e-4f + 1e-4f * std::fabs(a)) {
            std::cerr << "  identity " << name << ": " << a << " vs " << b << "\n";
            identityFails++;
        }
    };
    ident(host[PROBE_TRI_WI].w, host[PROBE_TRI_PDFS].y, "triSolidAnglePdf");
    ident(host[PROBE_ENVDIST_SAMPLE].z, host[PROBE_ENVDIST_SAMPLE].w, "envdistSamplePdf");
    ident(host[PROBE_ENV_PDF_DIR].x, host[PROBE_ENV_SAMPLE_DIR].w, "envDirPdf");
    ident(host[PROBE_ENV_PDF_DIR].z, 0.7f, "equirectRoundTripU");
    ident(host[PROBE_ENV_PDF_DIR].w, 0.4f, "equirectRoundTripV");
    ident(host[PROBE_ONE_SIDED_ZEROS].x, 0.0f, "backFaceSampleZero");
    ident(host[PROBE_ONE_SIDED_ZEROS].y, 0.0f, "backFacePdfZero");
    std::cout << (identityFails == 0 ? "PASS" : "FAIL") << " lightSamplePdfIdentities\n";

    try {
        // GPU personality (MSL on macOS, CUDA on Windows): same headers + the
        // same probe body, on the GPU. Metal compiles them at runtime; the CUDA
        // kernel is registered by shared_probe_kernels.cu.
        rhi::DeviceDesc desc;
        if (rhi::kNativeBackend == rhi::BackendKind::Metal) {
            desc.shaderSource = readTextFile(std::string(RHI_SHADER_DIR) + "/gpu_portable.h")
                              + "\n" + readTextFile(std::string(RHI_SHADER_DIR) + "/primitives_shared.h")
                              + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/spectrum_shared.h")
                              + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/bsdf_shared.h")
                              + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/envmap_shared.h")
                              + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/light_shared.h")
                              + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/tonemap_shared.h")
                              + "\n" + readTextFile(std::string(TEST_SHADER_DIR) + "/shared_probe.h");
        }
        auto device = rhi::createDevice(rhi::kNativeBackend, desc);
        auto stream = device->createStream();
        auto pipeline = device->createPipeline({ "shared_probe" });
        auto out = makeShared(*device, nullptr, kSlots * sizeof(float) * 4, "probe.out");
        auto r2sBuf = makeShared(*device, r2sHost.data(),
                                 r2sHost.size() * sizeof(float), "probe.r2s");
        auto spdBuf = makeShared(*device, tables.buffer().data(),
                                 tables.buffer().size() * sizeof(float), "probe.spd");
        auto xfBuf = makeShared(*device, xfHost, sizeof(xfHost), "probe.xf");
        auto envBuf = makeShared(*device, envDist.data(),
                                 envDist.size() * sizeof(float), "probe.env");
        stream->dispatch(*pipeline, { 1, 1, 1 }, { 32, 1, 1 }, nullptr, 0,
                         { out.get(), r2sBuf.get(), spdBuf.get(), xfBuf.get(), envBuf.get() });
        stream->waitIdle();

        const float* gpu = (const float*)out->hostPtr();
        int mismatches = 0;
        for (int i = 0; i < kSlots; i++) {
            for (int c = 0; c < 4; c++) {
                float h = host[i][c];
                float g = gpu[i * 4 + c];
                float tol = 2e-5f + 2e-5f * std::fabs(h);
                if (std::fabs(h - g) > tol) {
                    std::cerr << "  slot " << i << "[" << c << "]: host " << h
                              << " vs gpu " << g << "\n";
                    mismatches++;
                }
            }
        }
        std::cout << (mismatches == 0 ? "PASS" : "FAIL") << " sharedValueParity ("
                  << kSlots << " slots, host C++ vs " << rhi::backendName(rhi::kNativeBackend)
                  << ")\n";
        return (mismatches == 0 && identityFails == 0) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
