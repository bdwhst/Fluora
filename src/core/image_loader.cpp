#include "image_loader.h"

#include <cstring>

#include <stb_image.h>

bool loadHdrImage(const std::string& path, HdrImage& out)
{
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
