#pragma once
// Host-side scaffolding shared by the src/test executables: the PASS/FAIL
// check protocol (main returns nonzero iff `failures` != 0), text-file
// slurping for the runtime MSL concatenation, and Shared-buffer upload.
// Host C++ only — never part of a shader source concatenation.
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../rhi/rhi.h"

inline int failures = 0;

inline void check(bool ok, const char* name)
{
    std::cout << (ok ? "PASS " : "FAIL ") << name << "\n";
    if (!ok)
        failures++;
}

inline std::string readTextFile(const std::string& path)
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline std::unique_ptr<rhi::Buffer> makeShared(rhi::Device& dev, const void* data,
                                               size_t bytes, const char* name)
{
    auto buf = dev.createBuffer({ bytes, rhi::MemoryLocation::Shared, name });
    if (data)
        std::memcpy(buf->hostPtr(), data, bytes);
    else
        std::memset(buf->hostPtr(), 0, bytes);
    return buf;
}
