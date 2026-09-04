// White-furnace test (energy conservation + MIS correctness): a pure-white
// diffuse sphere in a uniform environment must disappear — every camera path
// either escapes directly or bounces off albedo-1 surfaces before escaping
// (the sphere is convex, so at most one bounce), and the estimator is
// unbiased, so sphere-covered pixels converge to the same value as
// background pixels. This catches integrator bugs the mega==wavefront cmp
// cannot: a wrong MIS weight, a lost or double-counted NEE term, or a
// cosine/pdf error shows up identically in both modes.
//
// Generates the scene (uniform 16x8 .hdr + a white sphere) in a per-process
// temp dir (removed on success, kept for inspection on failure — a fresh dir
// each run means a broken dump can never alias a previous run's data),
// drives the FluoraMini binary (expected next to this executable) headlessly
// with --safe-math, and reads the raw float accumulator via MINI_DUMP_ACCUM.
// Checks, per mode: all values finite, background nonzero, sphere-region
// mean within [0.97, 1.02] of the background mean per channel (rgb2spec
// "white" sits slightly below 1, so exact unity is not expected), and the
// mega and wavefront dumps bitwise identical (the --safe-math invariant,
// here covering the env-NEE path). Exits nonzero on failure.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr int kRes = 200;       // must match RES in the scene below
constexpr int kSpp = 256;
constexpr float kSphereR = 30;  // px: safely inside the ~44 px silhouette
constexpr float kBgR = 55;      // px: safely outside it

int failures = 0;

void check(bool ok, const char* name)
{
    std::cout << (ok ? "PASS " : "FAIL ") << name << "\n";
    if (!ok)
        failures++;
}

void setEnvVar(const char* name, const std::string& value)
{
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

int processId()
{
#ifdef _WIN32
    return _getpid();
#else
    return getpid();
#endif
}

// Uniform Radiance .hdr: flat (non-RLE) RGBE scanlines; (128,128,128,129)
// decodes to exactly 1.0 under stb (128 * 2^(129-136)).
void writeUniformHdr(const fs::path& path, int w, int h)
{
    std::ofstream f(path, std::ios::binary);
    f << "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y " << h << " +X " << w << "\n";
    const unsigned char texel[4] = { 128, 128, 128, 129 };
    for (int i = 0; i < w * h; i++)
        f.write((const char*)texel, 4);
}

void writeScene(const fs::path& path, const std::string& hdrName)
{
    std::ofstream f(path);
    f << "MATERIAL 0\n"
         "TYPE diffuse\n"
         "RGB         1 1 1\n"
         "ROUGHNESS   0.0\n"
         "METALLIC    0\n"
         "REFRIOR     0\n"
         "EMITTANCE   0\n"
         "\n"
         "CAMERA\n"
         "RES         " << kRes << " " << kRes << "\n"
         "FOVY        45\n"
         "ITERATIONS  " << kSpp << "\n"
         "DEPTH       8\n"
         "FILE        furnace\n"
         "EYE         0 0 5\n"
         "LOOKAT      0 0 0\n"
         "UP          0 1 0\n"
         "\n"
         "OBJECT 0\n"
         "geometry sphere\n"
         "material 0\n"
         "TRANS       0 0 0\n"
         "ROTAT       0 0 0\n"
         "SCALE       4 4 4\n"
         "\n"
         "SKYBOX\n"
      << hdrName << "\n";
}

bool runMode(const fs::path& exe, const fs::path& scene, const fs::path& dump, const char* mode)
{
    setEnvVar("MINI_DUMP_ACCUM", dump.string());
    std::string cmd = "\"" + exe.string() + "\" \"" + scene.string() + "\" --no-preview --spp "
                    + std::to_string(kSpp) + " --mode " + mode + " --safe-math --out \""
                    + dump.string() + ".png\"";
#ifdef _WIN32
    cmd = "\"" + cmd + "\"";  // cmd.exe strips the outer quote pair
#endif
    return std::system(cmd.c_str()) == 0;
}

std::vector<float> readDump(const fs::path& path)
{
    std::ifstream f(path, std::ios::binary);
    std::vector<float> v((size_t)kRes * kRes * 4);
    f.read((char*)v.data(), (std::streamsize)(v.size() * sizeof(float)));
    if (!f)
        v.clear();
    return v;
}

// Sphere-region vs background-region channel means; false on non-finite data.
bool analyze(const std::vector<float>& acc, const char* mode)
{
    double sphere[3] = {}, bg[3] = {};
    long nSphere = 0, nBg = 0;
    bool finite = true;
    for (int y = 0; y < kRes; y++) {
        for (int x = 0; x < kRes; x++) {
            const float* px = &acc[((size_t)y * kRes + x) * 4];
            for (int c = 0; c < 3; c++)
                finite = finite && std::isfinite(px[c]);
            float r = std::hypot(x - kRes * 0.5f, y - kRes * 0.5f);
            if (r < kSphereR) {
                for (int c = 0; c < 3; c++)
                    sphere[c] += px[c];
                nSphere++;
            } else if (r > kBgR) {
                for (int c = 0; c < 3; c++)
                    bg[c] += px[c];
                nBg++;
            }
        }
    }
    check(finite, (std::string("finite:") + mode).c_str());
    bool bgOk = true, ratioOk = true;
    for (int c = 0; c < 3; c++) {
        double bgMean = bg[c] / nBg;
        double ratio = (sphere[c] / nSphere) / bgMean;
        bgOk = bgOk && bgMean > 0.0;
        ratioOk = ratioOk && ratio > 0.97 && ratio < 1.02;
        std::cout << "  " << mode << " ch" << c << ": sphere/background = " << ratio << "\n";
    }
    check(bgOk, (std::string("backgroundNonzero:") + mode).c_str());
    check(ratioOk, (std::string("furnaceRatio:") + mode).c_str());
    return finite && bgOk && ratioOk;
}

} // namespace

int main(int, char** argv)
{
    fs::path exe = fs::path(argv[0]).parent_path() / "FluoraMini";
    fs::path dir = fs::temp_directory_path()
                 / ("fluora_furnace_" + std::to_string(processId()));
    fs::create_directories(dir);
    writeUniformHdr(dir / "furnace_env.hdr", 16, 8);
    writeScene(dir / "furnace.txt", "furnace_env.hdr");

    fs::path dumpWf = dir / "acc_wavefront.bin", dumpMega = dir / "acc_mega.bin";
    check(runMode(exe, dir / "furnace.txt", dumpWf, "wavefront"), "renderWavefront");
    check(runMode(exe, dir / "furnace.txt", dumpMega, "mega"), "renderMega");
    if (failures)
        return 1;

    std::vector<float> wf = readDump(dumpWf), mega = readDump(dumpMega);
    check(!wf.empty() && !mega.empty(), "accumDumpsRead");
    if (failures)
        return 1;

    analyze(wf, "wavefront");
    analyze(mega, "mega");
    check(std::memcmp(wf.data(), mega.data(), wf.size() * sizeof(float)) == 0,
          "megaWavefrontBitwise(safe math)");
    if (failures == 0)
        fs::remove_all(dir);
    else
        std::cerr << "artifacts kept in " << dir << "\n";
    return failures == 0 ? 0 : 1;
}
