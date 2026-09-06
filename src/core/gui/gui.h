#pragma once
// Portable overlay GUI for the Fluora renderers. This module issues only
// ImGui:: widget calls and reads input through ImGui's backend-neutral IO, so
// the same code drives the Metal (FluoraMini) and, later, the OpenGL/GLFW
// (Windows) previews — the backend wiring lives in each platform's RHI seam,
// never here (invariant: src/core is backend-neutral).
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace gui {

// A fly camera driven entirely by ImGui IO: left-drag looks, WASD moves in the
// view plane, E/C move along world up, the wheel dollies forward. update() is
// the only place that touches ImGui, so the type stays header-only-portable.
struct FlyCamera {
    glm::vec3 eye{ 0.0f, 0.0f, 0.0f };
    float yaw = 0.0f;    // radians about world up (+Y)
    float pitch = 0.0f;  // radians, clamped away from the poles
    float moveSpeed = 4.0f;    // units / second (WASD, E/C)
    float lookSens = 0.004f;   // radians / pixel of drag
    float dollyStep = 0.75f;   // units / wheel notch

    static FlyCamera fromLookAt(const glm::vec3& eye, const glm::vec3& lookAt);

    glm::vec3 forward() const;         // unit view direction from yaw/pitch
    glm::vec3 right() const;           // normalize(cross(forward, worldUp))
    glm::vec3 up() const;              // normalize(cross(right, forward))

    // Reads this frame's ImGui IO and applies motion; returns true when the view
    // changed (the caller should then restart accumulation). Honors
    // io.WantCaptureMouse / io.WantCaptureKeyboard so dragging a panel or typing
    // never moves the camera.
    bool update();
};

// Read-only figures the overlay displays; the app refills these each frame.
struct Stats {
    int sampleCount = 0;      // samples accumulated since the last reset
    int targetSpp = 0;
    double elapsedSec = 0.0;
    float samplesPerSec = 0.0f;
    std::string mode;         // "wavefront" / "mega"
    int numObjects = 0;
    size_t numTris = 0;
};

// Persistent UI state plus one-shot command flags the app reads back after
// draw() and then clears.
struct State {
    Stats stats;
    FlyCamera* camera = nullptr;   // the controls panel edits it live; may be null

    std::vector<std::string> sceneNames;  // dropdown labels (scene basenames)
    int selectedScene = 0;                // index currently loaded
    int requestedScene = -1;              // set by the combo; app consumes, resets to -1
    int maxDepth = 0;                     // live path-depth override; the app seeds it
                                          // from the scene and reads it every sample
    bool depthChanged = false;            // draw() sets it when the slider moved; the
                                          // loop restarts accumulation + clears
    bool saveRequested = false;           // app saves a PNG, then clears
    bool resetRequested = false;          // app zeroes accumulation, then clears
    bool cameraMoved = false;             // draw() sets it when the fly camera moved
                                          // this frame; app applies + clears
};

// Emits the whole overlay (stats, controls, scene picker) and, when `s.camera`
// is set, advances the fly camera from this frame's ImGui IO (setting
// s.cameraMoved on motion). Both the widgets and the input read happen here so
// every ImGui:: call stays inside one NewFrame()/Render() pair: call draw()
// between ImGui::NewFrame() and ImGui::Render(). Backend-agnostic.
void draw(State& s);

// Populates the scene dropdown from the `.txt` / `.json` files sibling to `scenePath`
// (sorted by name). `selected` is set to the entry matching `scenePath`, else 0.
// Never throws: on any filesystem error the outputs are left empty.
void scanSceneDirectory(const std::string& scenePath, std::vector<std::string>& names,
                        std::vector<std::string>& paths, int& selected);

// Renderer-specific steps the interactive loop drives. The loop never names a
// GPU/backend type — everything device-side happens in these hooks, so the same
// loop policy serves the Metal preview and the future OpenGL one. `samples` is
// the accumulation count the loop is tracking (for tonemap averaging / saving).
struct PreviewHooks {
    // Contract: hooks that read or clear the accumulation buffer host-side
    // (zeroAccum, save, finish, loadScene teardown) must drain in-flight GPU
    // work first — renderSample submits asynchronously and the loop never
    // waits on the device itself.
    std::function<void(int iter)> renderSample;      // accumulate one sample
    std::function<bool(int samples)> presentFrame;   // tonemap + blit; false = window closed
    std::function<void()> applyCamera;               // write State::camera into the renderer
    std::function<void()> zeroAccum;                 // drain + clear the accumulation buffer
    std::function<void(int sceneIdx)> loadScene;     // swap scene (camera reset lives here)
    std::function<void(int samples)> save;           // drain + write a PNG now
    std::function<void(int samples)> finish;         // final save on exit
};

// Drives the interactive preview until the window closes: renders samples up to
// `targetSpp`, presents at ~60 Hz, and restarts accumulation on camera move /
// reset / scene switch. Owns sample accounting, pacing, and command dispatch;
// all GPU work is in `hooks`. This is the reusable session policy that used to
// live in the app's main loop.
void runPreview(State& ui, int targetSpp, const PreviewHooks& hooks);

} // namespace gui
