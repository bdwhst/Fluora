#include "image_loader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <stb_image.h>
// OpenEXR via tinyexr, zlib from stb (stb.cpp compiles the stb
// implementations into every core-using target), as scene.cpp configures it.
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

namespace {

bool hasExtension(const std::string& path, const char* ext)
{
    if (path.size() < 4)
        return false;
    std::string tail = path.substr(path.size() - 4);
    for (auto& c : tail)
        c = (char)tolower((unsigned char)c);
    return tail == ext;
}

// tinyexr returns rows top-down; flip to the bottom-up order stbi's
// flip-on-load quirk gives every other image (see loadHdrImage).
bool loadExrBottomUp(const std::string& path, HdrImage& out)
{
    float* rgba = nullptr;
    int width = 0, height = 0;
    const char* err = nullptr;
    if (LoadEXR(&rgba, &width, &height, path.c_str(), &err) != TINYEXR_SUCCESS) {
        std::cout << "core: EXR load failed for " << path << ": " << (err ? err : "?") << "\n";
        if (err)
            FreeEXRErrorMessage(err);
        return false;
    }
    out.width = width;
    out.height = height;
    out.rgba.resize((size_t)width * height * 4);
    for (int y = 0; y < height; y++)
        std::memcpy(&out.rgba[(size_t)(height - 1 - y) * width * 4], &rgba[(size_t)y * width * 4],
                    (size_t)width * 4 * sizeof(float));
    std::free(rgba);
    return true;
}

} // namespace

bool loadHdrImage(const std::string& path, HdrImage& out)
{
    if (hasExtension(path, ".exr"))
        return loadExrBottomUp(path, out);
    // The CUDA renderer sets stbi_set_flip_vertically_on_load(1) globally
    // (main.cpp) and its texture lookups assume it — e.g. the equirectangular
    // v = asin(y)+0.5 mapping only puts the ceiling up on a bottom-up image.
    // Replicate the quirk so shared device code samples identically.
    stbi_set_flip_vertically_on_load(1);
    int width = 0, height = 0, channels = 0;
    float* data = stbi_loadf(path.c_str(), &width, &height, &channels, 4);
    if (!data)
        return false;
    out.width = width;
    out.height = height;
    out.rgba.resize((size_t)width * height * 4);
    std::memcpy(out.rgba.data(), data, out.rgba.size() * sizeof(float));
    stbi_image_free(data);
    return true;
}

bool loadLdrImage(const std::string& path, LdrImage& out)
{
    if (hasExtension(path, ".exr")) {
        // Linear float base color -> the sRGB-encoded 8-bit heap format the
        // material textures use (the sampler decodes it back). Loses the
        // float precision; a float texture format is the fix if it shows.
        HdrImage hdr;
        if (!loadExrBottomUp(path, hdr))
            return false;
        out.width = hdr.width;
        out.height = hdr.height;
        out.rgba.resize(hdr.rgba.size());
        for (size_t i = 0; i < hdr.rgba.size(); i++) {
            float v = std::clamp(hdr.rgba[i], 0.0f, 1.0f);
            if (i % 4 != 3)
                v = v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
            out.rgba[i] = (unsigned char)std::lround(v * 255.0f);
        }
        return true;
    }
    stbi_set_flip_vertically_on_load(1);  // same quirk as loadHdrImage
    int width = 0, height = 0, channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 4);
    if (!data)
        return false;
    out.width = width;
    out.height = height;
    out.rgba.assign(data, data + (size_t)width * height * 4);
    stbi_image_free(data);
    return true;
}
