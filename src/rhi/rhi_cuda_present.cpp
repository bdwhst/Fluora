// Presentation for the CUDA backend (see rhi_cuda_present.h). Mirrors the
// Metal backend's window semantics: the present target is an RGBA8 buffer the
// renderer's tonemap kernel writes (row 0 = top), present() blits it and
// composites the ImGui overlay, and q/Esc/close request exit.
#include "rhi_cuda_present.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <stdexcept>
#include <string>

#ifdef RHI_ENABLE_IMGUI
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#endif

namespace rhi {
namespace cuda {

namespace {

void cudaCheckP(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

// Fullscreen textured quad. The texture is uploaded with row 0 first, which
// GL treats as the bottom row, so the v coordinate is flipped here to put
// present-target row 0 at the top of the window. x is left as is: the tonemap
// kernel already mirrors it to match the saved-PNG convention.
const char* kVS =
    "#version 120\n"
    "attribute vec2 Position;\n"
    "attribute vec2 Texcoords;\n"
    "varying vec2 v_uv;\n"
    "void main() { v_uv = Texcoords; gl_Position = vec4(Position, 0.0, 1.0); }\n";
const char* kFS =
    "#version 120\n"
    "varying vec2 v_uv;\n"
    "uniform sampler2D u_image;\n"
    "void main() { gl_FragColor = texture2D(u_image, v_uv); }\n";

GLuint compileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024] = {};
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        throw std::runtime_error(std::string("present shader compile failed: ") + log);
    }
    return s;
}

} // namespace

struct Presenter::Impl {
    GLFWwindow* window = nullptr;
    GLuint tex = 0, pbo = 0, vao = 0, vbo[3] = { 0, 0, 0 }, program = 0;
    cudaGraphicsResource* pboRes = nullptr;
    int width = 0, height = 0;
    bool closeRequested = false;
    bool gui = false;
    const GuiDrawFn* guiDraw = nullptr;
};

Presenter::Presenter(int width, int height, const GuiDrawFn* gui) : mImpl(new Impl)
{
    Impl& im = *mImpl;
    im.width = width;
    im.height = height;
    im.gui = gui != nullptr;
    im.guiDraw = gui;

    if (!glfwInit())
        throw std::runtime_error("glfwInit failed");
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    im.window = glfwCreateWindow(width, height, "FluoraMini", nullptr, nullptr);
    if (!im.window) {
        glfwTerminate();
        throw std::runtime_error("glfwCreateWindow failed");
    }
    glfwMakeContextCurrent(im.window);
    // The preview loop paces presents itself; a blocking swap would throttle
    // render submission behind the display refresh.
    glfwSwapInterval(0);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
        throw std::runtime_error("glewInit failed");

    // Display texture + CUDA-registered PBO feeding it.
    glGenTextures(1, &im.tex);
    glBindTexture(GL_TEXTURE_2D, im.tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glGenBuffers(1, &im.pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, im.pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, (GLsizeiptr)width * height * 4, nullptr, GL_DYNAMIC_COPY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaCheckP(cudaGraphicsGLRegisterBuffer(&im.pboRes, im.pbo, cudaGraphicsRegisterFlagsWriteDiscard),
               "cudaGraphicsGLRegisterBuffer");

    // Quad: positions, texcoords (v flipped, see kVS), indices.
    const GLfloat verts[] = { -1, -1, 1, -1, 1, 1, -1, 1 };
    const GLfloat uvs[] = { 0, 1, 1, 1, 1, 0, 0, 0 };
    const GLushort idx[] = { 0, 1, 3, 3, 1, 2 };
    glGenVertexArrays(1, &im.vao);
    glBindVertexArray(im.vao);
    glGenBuffers(3, im.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, im.vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, im.vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, im.vbo[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
    glBindVertexArray(0);

    im.program = glCreateProgram();
    GLuint vs = compileShader(GL_VERTEX_SHADER, kVS);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFS);
    glAttachShader(im.program, vs);
    glAttachShader(im.program, fs);
    glBindAttribLocation(im.program, 0, "Position");
    glBindAttribLocation(im.program, 1, "Texcoords");
    glLinkProgram(im.program);
    GLint linked = 0;
    glGetProgramiv(im.program, GL_LINK_STATUS, &linked);
    if (!linked)
        throw std::runtime_error("present shader link failed");
    glDeleteShader(vs);
    glDeleteShader(fs);
    glUseProgram(im.program);
    glUniform1i(glGetUniformLocation(im.program, "u_image"), 0);

#ifdef RHI_ENABLE_IMGUI
    if (im.gui) {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui::GetIO().IniFilename = nullptr;  // no imgui.ini for a CLI tool
        ImGui_ImplGlfw_InitForOpenGL(im.window, true);
        ImGui_ImplOpenGL3_Init("#version 130");
    }
#endif
}

Presenter::~Presenter()
{
    Impl& im = *mImpl;
#ifdef RHI_ENABLE_IMGUI
    if (im.gui) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }
#endif
    if (im.pboRes)
        cudaGraphicsUnregisterResource(im.pboRes);
    if (im.window) {
        glDeleteProgram(im.program);
        glDeleteBuffers(3, im.vbo);
        glDeleteVertexArrays(1, &im.vao);
        glDeleteBuffers(1, &im.pbo);
        glDeleteTextures(1, &im.tex);
        glfwDestroyWindow(im.window);
    }
    glfwTerminate();
}

bool Presenter::present(const void* deviceRgba8)
{
    Impl& im = *mImpl;
    glfwPollEvents();
    if (glfwWindowShouldClose(im.window))
        return false;
    // Quit keys yield to ImGui while a widget owns the keyboard (Esc dismisses
    // popups / cancels slider text entry there).
    bool guiWantsKeys = false;
#ifdef RHI_ENABLE_IMGUI
    guiWantsKeys = im.gui && ImGui::GetIO().WantCaptureKeyboard;
#endif
    if (!guiWantsKeys
        && (glfwGetKey(im.window, GLFW_KEY_ESCAPE) == GLFW_PRESS
            || glfwGetKey(im.window, GLFW_KEY_Q) == GLFW_PRESS))
        return false;

    // Device-to-device copy into the mapped PBO on the legacy default stream:
    // blocking streams (every CommandStream) serialize with it, so the tonemap
    // dispatch that filled the target is complete before the copy reads it.
    void* pboPtr = nullptr;
    size_t pboBytes = 0;
    cudaCheckP(cudaGraphicsMapResources(1, &im.pboRes, 0), "cudaGraphicsMapResources");
    cudaCheckP(cudaGraphicsResourceGetMappedPointer(&pboPtr, &pboBytes, im.pboRes),
               "cudaGraphicsResourceGetMappedPointer");
    cudaCheckP(cudaMemcpyAsync(pboPtr, deviceRgba8, (size_t)im.width * im.height * 4,
                               cudaMemcpyDeviceToDevice, 0),
               "present copy");
    cudaCheckP(cudaGraphicsUnmapResources(1, &im.pboRes, 0), "cudaGraphicsUnmapResources");

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, im.pbo);
    glBindTexture(GL_TEXTURE_2D, im.tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, im.width, im.height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glViewport(0, 0, im.width, im.height);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(im.program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, im.tex);
    glBindVertexArray(im.vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);
    glBindVertexArray(0);

#ifdef RHI_ENABLE_IMGUI
    if (im.gui) {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        if (im.guiDraw && *im.guiDraw)
            (*im.guiDraw)();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
#endif
    glfwSwapBuffers(im.window);
    return true;
}

} // namespace cuda
} // namespace rhi
