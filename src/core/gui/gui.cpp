#include "gui.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <utility>

#include "imgui.h"

#include "../scene_loader.h"

namespace gui {

namespace {
const glm::vec3 kWorldUp{ 0.0f, 1.0f, 0.0f };
// Keep pitch just shy of vertical so forward() never becomes parallel to
// worldUp (which would collapse right()).
constexpr float kPitchLimit = 1.55334f;  // ~89 degrees

const char* materialTypeName(CoreMaterialType t)
{
    switch (t) {
    case CoreMaterialType::Diffuse: return "diffuse";
    case CoreMaterialType::Emissive: return "emissive";
    case CoreMaterialType::Dielectric: return "glass";
    case CoreMaterialType::Conductor: return "conductor";
    case CoreMaterialType::Interface: return "interface";
    }
    return "?";
}

// The "Materials" window: a swatch/name/type list of the scene's materials
// and, for the selection, per-type widgets editing the CoreMaterial in place.
// Any edit sets s.materialsChanged so the loop re-uploads and restarts
// accumulation (mixing samples of different materials would ghost).
void drawMaterials(State& s)
{
    if (!s.materials || s.materials->empty())
        return;
    std::vector<CoreMaterial>& mats = *s.materials;
    s.selectedMaterial = std::clamp(s.selectedMaterial, 0, (int)mats.size() - 1);

    ImGui::SetNextWindowPos(ImVec2(340, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 0), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Materials")) {
        // --- List ---------------------------------------------------------
        float rowH = ImGui::GetTextLineHeightWithSpacing();
        float listH = rowH * (float)std::min<size_t>(mats.size(), 8)
                      + ImGui::GetStyle().WindowPadding.y * 2.0f;
        ImGui::BeginChild("mat-list", ImVec2(0, listH), true);
        for (int i = 0; i < (int)mats.size(); i++) {
            const CoreMaterial& m = mats[i];
            ImGui::PushID(i);
            ImGui::ColorButton("##swatch", ImVec4(m.rgb.x, m.rgb.y, m.rgb.z, 1.0f),
                               ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop,
                               ImVec2(rowH - 4.0f, rowH - 4.0f));
            ImGui::SameLine();
            char label[128];
            std::snprintf(label, sizeof(label), "%s  [%s]",
                          m.name.empty() ? "(unnamed)" : m.name.c_str(),
                          materialTypeName(m.type));
            if (ImGui::Selectable(label, i == s.selectedMaterial))
                s.selectedMaterial = i;
            ImGui::PopID();
        }
        ImGui::EndChild();

        // --- Parameters for the selection ---------------------------------
        CoreMaterial& m = mats[s.selectedMaterial];
        bool edited = false;

        if (m.type == CoreMaterialType::Interface) {
            // Surfaceless medium boundaries: the type encodes the scene's
            // medium topology (mediumIn/mediumOut), so it is not switchable.
            ImGui::TextDisabled("medium boundary (in %d / out %d) - not editable",
                                m.mediumIn, m.mediumOut);
        } else {
            static const CoreMaterialType kTypes[] = {
                CoreMaterialType::Diffuse, CoreMaterialType::Emissive,
                CoreMaterialType::Dielectric, CoreMaterialType::Conductor
            };
            if (ImGui::BeginCombo("type", materialTypeName(m.type))) {
                for (CoreMaterialType t : kTypes) {
                    if (ImGui::Selectable(materialTypeName(t), t == m.type) && t != m.type) {
                        m.type = t;
                        // Named spectra were bound to the original type (a
                        // glass eta makes no sense on a conductor and vice
                        // versa); a switched material runs on constant ior /
                        // reflectance rgb. The upload honors the same rule.
                        m.etaNamed.clear();
                        m.kNamed.clear();
                        // A light with zero emittance renders black; seed
                        // something visible on the first switch.
                        if (t == CoreMaterialType::Emissive && m.emittance <= 0.0f)
                            m.emittance = 1.0f;
                        edited = true;
                    }
                }
                ImGui::EndCombo();
            }
            switch (m.type) {
            case CoreMaterialType::Diffuse:
                edited |= ImGui::ColorEdit3("albedo", &m.rgb.x);
                if (m.texIdx != kCoreTexNone)
                    ImGui::TextDisabled("tints the base-color texture");
                break;
            case CoreMaterialType::Emissive:
                edited |= ImGui::ColorEdit3("color", &m.rgb.x);
                edited |= ImGui::DragFloat("emittance", &m.emittance, 0.05f, 0.0f, 1e6f,
                                           "%.2f", ImGuiSliderFlags_Logarithmic);
                break;
            case CoreMaterialType::Dielectric:
                if (!m.etaNamed.empty())
                    ImGui::TextDisabled("dispersive eta: %s", m.etaNamed.c_str());
                else
                    edited |= ImGui::SliderFloat("ior", &m.ior, 1.0f, 2.5f, "%.3f",
                                                 ImGuiSliderFlags_AlwaysClamp);
                break;
            case CoreMaterialType::Conductor:
                edited |= ImGui::SliderFloat("roughness", &m.roughness, 0.0f, 1.0f, "%.4f",
                                             ImGuiSliderFlags_Logarithmic);
                if (m.roughness < 1e-3f) {
                    ImGui::SameLine();
                    ImGui::TextDisabled("(mirror)");
                }
                // The upload needs BOTH measured spectra, else it falls back
                // to reflectance mode driven by rgb (mini_main's conversion).
                if (!m.etaNamed.empty() && !m.kNamed.empty())
                    ImGui::TextDisabled("measured eta/k: %s", m.etaNamed.c_str());
                else
                    edited |= ImGui::ColorEdit3("reflectance", &m.rgb.x);
                break;
            default:
                break;
            }
        }
        if (edited)
            s.materialsChanged = true;
    }
    ImGui::End();
}
} // namespace

FlyCamera FlyCamera::fromLookAt(const glm::vec3& eye, const glm::vec3& lookAt)
{
    FlyCamera c;
    c.eye = eye;
    glm::vec3 f = lookAt - eye;
    float len = glm::length(f);
    f = len > 1e-6f ? f / len : glm::vec3(0.0f, 0.0f, -1.0f);
    // Clamp like update() does: a scene camera looking straight up/down would
    // otherwise make forward() parallel to worldUp and right() NaN.
    c.pitch = std::clamp(std::asin(std::clamp(f.y, -1.0f, 1.0f)), -kPitchLimit, kPitchLimit);
    c.yaw = std::atan2(f.x, f.z);
    return c;
}

glm::vec3 FlyCamera::forward() const
{
    float cp = std::cos(pitch);
    return glm::vec3(cp * std::sin(yaw), std::sin(pitch), cp * std::cos(yaw));
}

glm::vec3 FlyCamera::right() const
{
    return glm::normalize(glm::cross(forward(), kWorldUp));
}

glm::vec3 FlyCamera::up() const
{
    return glm::normalize(glm::cross(right(), forward()));
}

bool FlyCamera::update()
{
    ImGuiIO& io = ImGui::GetIO();
    bool moved = false;

    // Look / dolly — ignored while the pointer is over a panel.
    if (!io.WantCaptureMouse) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            // Screen-up (MouseDelta.y < 0) tilts the view up; drag-right yaws.
            yaw -= io.MouseDelta.x * lookSens;
            pitch -= io.MouseDelta.y * lookSens;
            pitch = std::clamp(pitch, -kPitchLimit, kPitchLimit);
            if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f)
                moved = true;
        }
        if (io.MouseWheel != 0.0f) {
            eye += forward() * (io.MouseWheel * dollyStep);
            moved = true;
        }
    }

    // WASD / E / C — ignored while a widget owns the keyboard.
    if (!io.WantCaptureKeyboard) {
        float step = moveSpeed * io.DeltaTime;
        glm::vec3 f = forward(), r = right();
        glm::vec3 delta(0.0f);
        if (ImGui::IsKeyDown(ImGuiKey_W)) delta += f;
        if (ImGui::IsKeyDown(ImGuiKey_S)) delta -= f;
        if (ImGui::IsKeyDown(ImGuiKey_D)) delta += r;
        if (ImGui::IsKeyDown(ImGuiKey_A)) delta -= r;
        if (ImGui::IsKeyDown(ImGuiKey_E)) delta += kWorldUp;
        if (ImGui::IsKeyDown(ImGuiKey_C)) delta -= kWorldUp;
        if (delta != glm::vec3(0.0f)) {
            eye += delta * step;
            moved = true;
        }
    }
    return moved;
}

void draw(State& s)
{
    // Advance the fly camera from this frame's IO here, inside the caller's
    // NewFrame()/Render() pair, so update()'s ImGui queries never run on an
    // ended frame. Outside the Begin() below so a collapsed panel doesn't
    // freeze the camera. The app reads s.cameraMoved after the frame and
    // restarts accumulation.
    if (s.camera && s.camera->update())
        s.cameraMoved = true;

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 0), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Fluora")) {
        const Stats& st = s.stats;

        // --- Render stats -------------------------------------------------
        ImGui::Text("%s  |  %d objects, %zu tris", st.mode.c_str(), st.numObjects,
                    st.numTris);
        if (st.targetSpp > 0)
            ImGui::Text("samples: %d / %d", st.sampleCount, st.targetSpp);
        else
            ImGui::Text("samples: %d", st.sampleCount);
        ImGui::Text("%.1f spp/s   |   %.1f s", st.samplesPerSec, st.elapsedSec);
        // Live edit: mixing samples of different depths would bias the image,
        // so the loop restarts accumulation on change (same as camera motion).
        if (ImGui::SliderInt("max depth", &s.maxDepth, 1, 32, "%d",
                             ImGuiSliderFlags_AlwaysClamp))
            s.depthChanged = true;

        // --- Scene picker -------------------------------------------------
        ImGui::Separator();
        if (!s.sceneNames.empty()) {
            int shown = std::clamp(s.selectedScene, 0, (int)s.sceneNames.size() - 1);
            const char* preview = s.sceneNames[shown].c_str();
            if (ImGui::BeginCombo("scene", preview)) {
                for (int i = 0; i < (int)s.sceneNames.size(); i++) {
                    bool sel = (i == shown);
                    if (ImGui::Selectable(s.sceneNames[i].c_str(), sel) && i != s.selectedScene)
                        s.requestedScene = i;
                    if (sel)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }

        // --- Camera controls ---------------------------------------------
        ImGui::Separator();
        if (s.camera) {
            FlyCamera& c = *s.camera;
            ImGui::Text("eye  %.2f  %.2f  %.2f", c.eye.x, c.eye.y, c.eye.z);
            ImGui::SliderFloat("move speed", &c.moveSpeed, 0.1f, 50.0f, "%.1f",
                               ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("look sens", &c.lookSens, 0.001f, 0.02f, "%.4f");
            ImGui::TextDisabled("WASD move, E/C up-down, drag look, wheel dolly");
        }

        // --- Actions ------------------------------------------------------
        ImGui::Separator();
        if (ImGui::Button("Save PNG"))
            s.saveRequested = true;
        ImGui::SameLine();
        if (ImGui::Button("Reset accumulation"))
            s.resetRequested = true;
    }
    ImGui::End();

    drawMaterials(s);
}

void scanSceneDirectory(const std::string& scenePath, std::vector<std::string>& names,
                        std::vector<std::string>& paths, int& selected)
{
    namespace fs = std::filesystem;
    names.clear();
    paths.clear();
    selected = 0;
    try {
        fs::path here = fs::weakly_canonical(scenePath);
        std::vector<std::pair<std::string, std::string>> entries;  // (name, path)
        for (const auto& e : fs::directory_iterator(here.parent_path())) {
            if (e.is_regular_file()
                && (e.path().extension() == ".txt" || e.path().extension() == ".json"))
                entries.emplace_back(e.path().filename().string(), e.path().string());
        }
        std::sort(entries.begin(), entries.end());
        for (size_t i = 0; i < entries.size(); i++) {
            names.push_back(entries[i].first);
            paths.push_back(entries[i].second);
            if (fs::weakly_canonical(entries[i].second) == here)
                selected = (int)i;
        }
    } catch (const std::exception&) {
        names.clear();
        paths.clear();
        selected = 0;
    }
}

void runPreview(State& ui, int targetSpp, const PreviewHooks& h)
{
    using clock = std::chrono::steady_clock;
    auto lastPresent = clock::now() - std::chrono::seconds(1);
    auto resetTime = clock::now();
    int sampleCount = 0;  // samples accumulated since the last reset

    auto restart = [&] {
        h.zeroAccum();
        sampleCount = 0;
        resetTime = clock::now();
    };

    while (true) {
        bool rendered = false;
        if (sampleCount < targetSpp) {
            h.renderSample(sampleCount);
            sampleCount++;
            rendered = true;
            // Write the PNG the moment the target is reached (the window then
            // freezes at the final frame) so a long render isn't lost if the
            // app dies before the user closes the window; finish() re-saves on
            // close, capturing any post-convergence camera motion.
            if (sampleCount == targetSpp)
                h.save(sampleCount);
        }

        auto now = clock::now();
        // Present when a sample was throttled (converged) or ~60 Hz has elapsed.
        if (!rendered || now - lastPresent >= std::chrono::milliseconds(16)) {
            double elapsed = std::chrono::duration<double>(now - resetTime).count();
            ui.stats.sampleCount = sampleCount;
            ui.stats.elapsedSec = elapsed;
            ui.stats.samplesPerSec = elapsed > 0.0 ? (float)(sampleCount / elapsed) : 0.0f;
            lastPresent = now;

            if (!h.presentFrame(std::max(sampleCount, 1)))
                break;

            // present() just pumped events and built the ImGui frame (draw()
            // advanced the camera inside it); consume this frame's command
            // flags. Save first: it must capture the accumulation the user
            // just saw, before any same-frame restart zeroes it (holding W
            // while clicking "Save PNG" would otherwise write a black image).
            if (ui.saveRequested) {
                ui.saveRequested = false;
                h.save(sampleCount);
            }
            if (ui.cameraMoved) {
                ui.cameraMoved = false;
                h.applyCamera();
                restart();
            }
            if (ui.resetRequested) {
                ui.resetRequested = false;
                restart();
            }
            if (ui.depthChanged) {
                ui.depthChanged = false;
                restart();
            }
            if (ui.materialsChanged) {
                ui.materialsChanged = false;
                if (h.applyMaterials)
                    h.applyMaterials();
                restart();
            }
            if (ui.requestedScene >= 0) {
                int idx = ui.requestedScene;
                ui.requestedScene = -1;
                h.loadScene(idx);
                restart();
            }
            if (!rendered)  // converged: idle at the same ~60 Hz as rendering
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
    }
    h.finish(sampleCount);
}

} // namespace gui
