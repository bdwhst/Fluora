#pragma once
// Presentation half of the CUDA backend: a GLFW window with an OpenGL texture
// fed through a CUDA-registered PBO (the interop main.cpp/preview.cpp use),
// plus the optional Dear ImGui overlay (GLFW + OpenGL3 backends). Host-only
// C++ so nvcc never sees GL/GLFW/ImGui headers; rhi_cuda.cu owns one of these
// behind Device::presentTarget()/present().
#include <cuda_runtime.h>

#include <functional>
#include <memory>

namespace rhi {
namespace cuda {

class Presenter {
public:
    using GuiDrawFn = std::function<void()>;

    // Creates the window + GL resources; `gui` non-null enables the overlay.
    Presenter(int width, int height, const GuiDrawFn* gui);
    ~Presenter();
    Presenter(const Presenter&) = delete;
    Presenter& operator=(const Presenter&) = delete;

    // Pumps events, copies the RGBA8 present target (device pointer, row 0 =
    // top) into the window, draws the overlay, swaps. The copy is issued on the
    // legacy default stream so it orders after every blocking-stream submit.
    // Returns false once the user asked to close (window close, q, Esc).
    bool present(const void* deviceRgba8);

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cuda
} // namespace rhi
