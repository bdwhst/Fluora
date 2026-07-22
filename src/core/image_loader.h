#pragma once
// Host-side image loading for texture uploads (portable, no GPU deps).
#include <string>
#include <vector>

struct HdrImage {
    int width = 0;
    int height = 0;
    std::vector<float> rgba;  // width * height * 4, row-major, bottom row
                              // first (stbi flip quirk — see the .cpp)
};

// Loads a Radiance .hdr file as RGBA32F (stb_image). Returns false with the
// image untouched on failure.
bool loadHdrImage(const std::string& path, HdrImage& out);

struct LdrImage {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> rgba;  // width * height * 4, bottom row first
                                      // (same stbi flip quirk); sRGB-encoded
};

// Loads an 8-bit image (png/jpg/...) as RGBA8. Returns false on failure.
bool loadLdrImage(const std::string& path, LdrImage& out);
