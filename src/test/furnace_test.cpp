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
//
// A second scene runs the same check on participating media: a purely
// scattering (sigma_a = 0), chromatic, anisotropic homogeneous medium in a
// surfaceless box. Every scattering event conserves energy, so the box must
// vanish against the uniform environment just like the sphere does. This
// covers the delta-tracked distance sampling, HG phase sampling, the
// interface pass-through (camera paths and shadow rays), Beer-Lambert
// transmittance on shadow rays, and — because sigma_s differs per wavelength —
// the hero-wavelength spectral MIS: a wrong r ratio tints the box.
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

#include "test_util.h"

namespace fs = std::filesystem;

namespace {

constexpr int kRes = 200;       // must match RES in the scene below
constexpr int kSpp = 256;
constexpr float kSphereR = 30;  // px: safely inside the ~44 px silhouette
constexpr float kBgR = 55;      // px: safely outside it
constexpr float kBoxR = 15;     // px: inside the medium box's ~25 px near face
constexpr float kBoxBgR = 45;   // px: outside its ~35 px diagonal

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

// The medium scene (.json): a 2-unit box of scattering-only fog, camera in
// vacuum at z=5 looking at it. Optical depth across the box is 0.4..1.6
// depending on wavelength; DEPTH is generous so truncated paths lose no
// measurable energy.
void writeMediumScene(const fs::path& path, const std::string& hdrName)
{
    std::ofstream f(path);
    f << "{\n"
         "  \"Camera\": { \"RES\": [" << kRes << ", " << kRes << "], \"FOVY\": 45.0,\n"
         "    \"ITERATIONS\": " << kSpp << ", \"DEPTH\": 64, \"FILE\": \"furnace_medium\",\n"
         "    \"EYE\": [0, 0, 5], \"LOOKAT\": [0, 0, 0], \"UP\": [0, 1, 0] },\n"
         "  \"Background\": { \"TYPE\": \"skybox\", \"PATH\": \"" << hdrName << "\" },\n"
         "  \"Materials\": {},\n"
         "  \"Media\": {\n"
         "    \"fog\": { \"TYPE\": \"homogeneous\",\n"
         "      \"SIGMA_A\": { \"TYPE\": \"rgb\", \"VALUE\": [0, 0, 0] },\n"
         "      \"SIGMA_S\": { \"TYPE\": \"rgb\", \"VALUE\": [0.2, 0.4, 0.8] },\n"
         "      \"SIGMA_SCALE\": 1.0, \"G\": 0.3,\n"
         "      \"TRANS\": [0, 0, 0], \"ROTAT\": [0, 0, 0], \"SCALE\": [1, 1, 1] }\n"
         "  },\n"
         "  \"MediumInterfaces\": { \"fogBox\": { \"INSIDE\": \"fog\", \"OUTSIDE\": \"\" } },\n"
         "  \"Objects\": [\n"
         "    { \"TYPE\": \"model_inline\",\n"
         "      \"VERTICES\": [1,-1,1, -1,-1,1, 1,1,1, -1,1,1, -1,-1,-1, 1,-1,-1, -1,1,-1, 1,1,-1],\n"
         "      \"INDICES\": [0,3,1, 0,2,3, 4,7,5, 4,6,7, 6,2,7, 6,3,2, 5,1,4, 5,0,1, 5,2,0, 5,7,2, 1,6,4, 1,3,6],\n"
         "      \"MEDIUM_INTERFACE\": \"fogBox\",\n"
         "      \"TRANS\": [0, 0, 0], \"ROTAT\": [0, 0, 0], \"SCALE\": [1, 1, 1] }\n"
         "  ]\n"
         "}\n";
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

// Object-region (r < rIn) vs background-region (r > rOut) channel means;
// false on non-finite data.
bool analyze(const std::vector<float>& acc, const char* mode, float rIn, float rOut)
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
            if (r < rIn) {
                for (int c = 0; c < 3; c++)
                    sphere[c] += px[c];
                nSphere++;
            } else if (r > rOut) {
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
        std::cout << "  " << mode << " ch" << c << ": object/background = " << ratio << "\n";
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
    writeMediumScene(dir / "furnace_medium.json", "furnace_env.hdr");

    struct Case { const char* name; const char* scene; float rIn, rOut; };
    const Case cases[] = {
        { "sphere", "furnace.txt", kSphereR, kBgR },
        { "medium", "furnace_medium.json", kBoxR, kBoxBgR },
    };
    for (const Case& cs : cases) {
        std::cout << "== " << cs.name << " (" << cs.scene << ")\n";
        std::string tag = std::string(":") + cs.name;
        fs::path dumpWf = dir / (std::string("acc_wavefront_") + cs.name + ".bin");
        fs::path dumpMega = dir / (std::string("acc_mega_") + cs.name + ".bin");
        check(runMode(exe, dir / cs.scene, dumpWf, "wavefront"), ("renderWavefront" + tag).c_str());
        check(runMode(exe, dir / cs.scene, dumpMega, "mega"), ("renderMega" + tag).c_str());
        if (failures)
            break;

        std::vector<float> wf = readDump(dumpWf), mega = readDump(dumpMega);
        check(!wf.empty() && !mega.empty(), ("accumDumpsRead" + tag).c_str());
        if (failures)
            break;

        analyze(wf, ("wavefront" + tag).c_str(), cs.rIn, cs.rOut);
        analyze(mega, ("mega" + tag).c_str(), cs.rIn, cs.rOut);
        check(std::memcmp(wf.data(), mega.data(), wf.size() * sizeof(float)) == 0,
              ("megaWavefrontBitwise(safe math)" + tag).c_str());
    }
    if (failures == 0)
        fs::remove_all(dir);
    else
        std::cerr << "artifacts kept in " << dir << "\n";
    return failures == 0 ? 0 : 1;
}
