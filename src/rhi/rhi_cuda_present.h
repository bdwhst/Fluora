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

// Largest window *content* size the primary monitor can show: its work area
// (taskbar/dock excluded) minus an allowance for the window frame, which GLFW
// can only measure on a window that already exists. False when there is no
// monitor to ask (no display, GLFW init failure). Initializes GLFW if it is not
// up yet, so call it only when a window is about to be created.
bool displayContentSize(int& width, int& height);

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

    // Resizes the window and its display texture / interop PBO to a new
    // render size (the window itself is fixed-size for the user). Waits for
    // the in-flight present copy before re-registering the PBO.
    void resize(int width, int height);

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace cuda
} // namespace rhi
