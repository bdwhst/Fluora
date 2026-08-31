// SharedHostTest (docs/portable-device-code.md §3): proves the shared device
// code compiles in the non-Metal personality's spelling (this file compiles it
// as host C++ — the same personality nvcc sees modulo qualifiers) AND agrees
// numerically with the MSL personality: the same headers plus the same probe
// body (shared_probe.h) are compiled by the Metal compiler at runtime and
// evaluated on the GPU. Tolerance-based compare (Metal builds with fast-math).
// Exits nonzero on failure.
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "gpu_portable.h"
#include "../core/bsdf_shared.h"
#include "../core/envmap_shared.h"
#include "../core/tonemap_shared.h"
#include "shared_probe.h"

#include "rhi.h"

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

constexpr int kSlots = PROBE_SLOTS;

const char* kProbeKernel = R"MSL(
kernel void shared_probe(device float4* outv [[buffer(1)]],
                         uint tid [[thread_position_in_grid]])
{
    if (tid != 0)
        return;
    PROBE_BODY(outv)
}
)MSL";

} // namespace

int main()
{
    // Host personality evaluation.
    gpu_float4 host[kSlots] = {};
    PROBE_BODY(host)

    try {
        // MSL personality: same headers + the same probe body, on the GPU.
        rhi::DeviceDesc desc;
        desc.shaderSource = readTextFile(std::string(RHI_SHADER_DIR) + "/gpu_portable.h")
                          + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/bsdf_shared.h")
                          + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/envmap_shared.h")
                          + "\n" + readTextFile(std::string(CORE_SHADER_DIR) + "/tonemap_shared.h")
                          + "\n" + readTextFile(std::string(RHI_SHADER_DIR) + "/shared_probe.h")
                          + "\n" + kProbeKernel;
        auto device = rhi::createDevice(rhi::BackendKind::Metal, desc);
        auto stream = device->createStream();
        auto pipeline = device->createPipeline({ "shared_probe" });
        auto out = device->createBuffer(
            { kSlots * sizeof(float) * 4, rhi::MemoryLocation::Shared, "probe.out" });
        stream->dispatch(*pipeline, { 1, 1, 1 }, { 32, 1, 1 }, nullptr, 0, { out.get() });
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
                  << kSlots << " slots, host C++ vs MSL)\n";
        return mismatches == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
