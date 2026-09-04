// Unit tests for the RHI parallel primitives (design doc M2): reduce, scan,
// compact, radix sort, and the simdgroup-aggregated work queue, each checked
// against a CPU reference on random data. Exits nonzero on failure.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "../rhi/rhi.h"
#include "../rhi/rhi_algorithms.h"
#include "../rhi/primitives_shared.h"
#include "test_util.h"

int main()
{
    try {
        rhi::DeviceDesc desc;
        // Metal compiles the primitives + this test's kernel from source at
        // runtime; the CUDA backend has them pre-registered (rhi_cuda.cu,
        // rhi_test_kernels.cu).
        if (rhi::kNativeBackend == rhi::BackendKind::Metal) {
            desc.shaderSource = readTextFile(std::string(RHI_SHADER_DIR) + "/gpu_portable.h")
                              + "\n" + readTextFile(std::string(RHI_SHADER_DIR) + "/primitives_shared.h")
                              + "\n" + readTextFile(std::string(RHI_SHADER_DIR) + "/primitives_gpu.h")
                              + "\n" + readTextFile(std::string(RHI_SHADER_DIR) + "/texture_gpu.h")
                              + "\n" + readTextFile(std::string(TEST_SHADER_DIR) + "/rhi_test_gpu.h");
        }
        auto device = rhi::createDevice(rhi::kNativeBackend, desc);
        std::cout << "backend: " << rhi::backendName(rhi::kNativeBackend) << "\n";
        auto stream = device->createStream();
        rhi::Algorithms alg(*device);

        std::mt19937 rng(12345);
        const uint32_t n = 1'000'037;  // deliberately not a tile multiple
        std::vector<uint32_t> data(n);
        for (auto& v : data)
            v = rng();

        // reduce
        {
            auto in = makeShared(*device, data.data(), n * 4, "in");
            auto result = makeShared(*device, nullptr, 4, "result");
            alg.reduceSum(*stream, *in, n, *result);
            stream->waitIdle();
            uint32_t expect = 0;
            for (uint32_t v : data)
                expect += v;  // uint32 wraparound matches the GPU
            check(*(uint32_t*)result->hostPtr() == expect, "reduceSum");
        }

        // exclusive scan
        {
            auto in = makeShared(*device, data.data(), n * 4, "in");
            auto out = makeShared(*device, nullptr, n * 4, "out");
            alg.exclusiveScan(*stream, *in, n, *out);
            stream->waitIdle();
            std::vector<uint32_t> expect(n);
            uint32_t acc = 0;
            for (uint32_t i = 0; i < n; i++) {
                expect[i] = acc;
                acc += data[i];
            }
            check(std::memcmp(out->hostPtr(), expect.data(), n * 4) == 0, "exclusiveScan");
        }

        // compact
        {
            std::vector<uint32_t> flags(n);
            for (uint32_t i = 0; i < n; i++)
                flags[i] = (data[i] % 3 == 0) ? 1u : 0u;
            auto in = makeShared(*device, data.data(), n * 4, "in");
            auto flagBuf = makeShared(*device, flags.data(), n * 4, "flags");
            auto out = makeShared(*device, nullptr, n * 4, "out");
            uint32_t count = alg.compact(*stream, *in, *flagBuf, n, *out);
            std::vector<uint32_t> expect;
            for (uint32_t i = 0; i < n; i++)
                if (flags[i])
                    expect.push_back(data[i]);
            bool ok = count == expect.size()
                   && std::memcmp(out->hostPtr(), expect.data(), expect.size() * 4) == 0;
            check(ok, "compact");
        }

        // radix sort
        {
            const uint32_t ns = (1u << 20) + 13;
            std::vector<uint32_t> keys(ns);
            for (auto& v : keys)
                v = rng();
            auto buf = makeShared(*device, keys.data(), ns * 4, "keys");
            alg.radixSort(*stream, *buf, ns);
            stream->waitIdle();
            std::sort(keys.begin(), keys.end());
            check(std::memcmp(buf->hostPtr(), keys.data(), ns * 4) == 0, "radixSort");
        }

        // work queue (order is nondeterministic: compare as multisets)
        {
            auto pipeline = device->createPipeline({ "prim_queue_push_test" });
            auto in = makeShared(*device, data.data(), n * 4, "in");
            auto counter = makeShared(*device, nullptr, 4, "counter");
            auto out = makeShared(*device, nullptr, n * 4, "out");
            PrimParams p = {};
            p.n = n;
            stream->dispatch(*pipeline, { (n + PRIM_TILE - 1) / PRIM_TILE, 1, 1 },
                             { PRIM_TILE, 1, 1 }, &p, sizeof(p),
                             { in.get(), counter.get(), out.get() });
            stream->waitIdle();
            uint32_t count = *(uint32_t*)counter->hostPtr();
            std::vector<uint32_t> expect;
            for (uint32_t v : data)
                if (v & 1u)
                    expect.push_back(v);
            std::vector<uint32_t> got((uint32_t*)out->hostPtr(), (uint32_t*)out->hostPtr() + count);
            std::sort(got.begin(), got.end());
            std::sort(expect.begin(), expect.end());
            check(count == expect.size() && got == expect, "workQueuePush");
        }

        // bindless texture heap: bilinear sampling of known texels
        {
            // Texture 0: 2x2 RGBA32F with distinct texel colors.
            const float texA[2 * 2 * 4] = {
                1, 0, 0, 1,   0, 1, 0, 1,
                0, 0, 1, 1,   1, 1, 0, 1,
            };
            auto t0 = device->createTexture({ 2, 2, rhi::TextureFormat::RGBA32Float });
            t0->upload(texA, sizeof(texA));
            // Texture 1: 1x1 constant, verifies heap index routing.
            const float texB[4] = { 0.25f, 0.5f, 0.75f, 1 };
            auto t1 = device->createTexture({ 1, 1, rhi::TextureFormat::RGBA32Float });
            t1->upload(texB, sizeof(texB));

            // (u, v, heapIndex): texel centers reproduce exact texels under
            // bilinear; the center of the 2x2 blends all four equally.
            const float queries[][4] = {
                { 0.25f, 0.25f, 0, 0 }, { 0.75f, 0.25f, 0, 0 },
                { 0.25f, 0.75f, 0, 0 }, { 0.75f, 0.75f, 0, 0 },
                { 0.50f, 0.50f, 0, 0 }, { 0.33f, 0.77f, 1, 0 },
            };
            const float expect[][4] = {
                { 1, 0, 0, 1 }, { 0, 1, 0, 1 },
                { 0, 0, 1, 1 }, { 1, 1, 0, 1 },
                { 0.5f, 0.5f, 0.25f, 1 }, { 0.25f, 0.5f, 0.75f, 1 },
            };
            const uint32_t nq = 6;
            auto qBuf = makeShared(*device, queries, sizeof(queries), "queries");
            auto oBuf = makeShared(*device, nullptr, sizeof(expect), "out");
            auto pipeline = device->createPipeline({ "test_tex_sample" });
            PrimParams p = {};
            p.n = nq;
            stream->dispatch(*pipeline, { 1, 1, 1 }, { PRIM_TILE, 1, 1 }, &p, sizeof(p),
                             { &device->textureHeap(), qBuf.get(), oBuf.get() });
            stream->waitIdle();
            const float* got = (const float*)oBuf->hostPtr();
            bool ok = true;
            for (uint32_t i = 0; i < nq * 4; i++)
                ok = ok && std::abs(got[i] - ((const float*)expect)[i]) < 1e-5f;
            check(ok, "textureHeapSample");
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
