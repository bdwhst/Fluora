//#define _CRT_SECURE_NO_DEPRECATE
#include <ctime>
#include "main.h"
#include "preview.h"
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"
GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;
extern uint64_t sysTime;
extern uint64_t delta_t;

float camera_speed = 0.2f;

GLFWwindow* window;
GuiDataContainer* imguiData = NULL;
ImGuiIO* io = nullptr;
bool mouseOverImGuiWinow = false;

extern RenderState* renderState;
extern float zoom, theta, phi;
extern bool camchanged;


std::string currentTimeString() {
	time_t now;
	time(&now);
	char buf[sizeof "0000-00-00_00-00-00z"];
	strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
	return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures() {
	glGenTextures(1, &displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void) {
	GLfloat vertices[] = {
		-1.0f, -1.0f,
		1.0f, -1.0f,
		1.0f,  1.0f,
		-1.0f,  1.0f,
	};

	GLfloat texcoords[] = {
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

	GLuint vertexBufferObjID[3];
	glGenBuffers(3, vertexBufferObjID);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(positionLocation);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(texcoordsLocation);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader() {
	const char* attribLocations[] = { "Position", "Texcoords" };
	GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
	GLint location;

	//glUseProgram(program);
	if ((location = glGetUniformLocation(program, "u_image")) != -1) {
		glUniform1i(location, 0);
	}

	return program;
}

void deletePBO(GLuint* pbo) {
	if (pbo) {
		// unregister this buffer object with CUDA
		cudaGLUnregisterBufferObject(*pbo);

		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);

		*pbo = (GLuint)NULL;
	}
}

void deleteTexture(GLuint* tex) {
	glDeleteTextures(1, tex);
	*tex = (GLuint)NULL;
}

void cleanupCuda() {
	if (pbo) {
		deletePBO(&pbo);
	}
	if (displayImage) {
		deleteTexture(&displayImage);
	}
}

void initCuda() {
	cudaDeviceReset();
	cudaGLSetGLDevice(0);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		exit(-1);
	}

	// Clean up on program exit
	atexit(cleanupCuda);
}

void initPBO() {
	// set up vertex data parameter
	int num_texels = width * height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;

	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	glGenBuffers(1, &pbo);

	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject(pbo);

}

void errorCallback(int error, const char* description) {
	fprintf(stderr, "%s\n", description);
}

static ImFont* appFont = nullptr;


bool init() {
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}

	window = glfwCreateWindow(width, height, "Fluora", NULL, NULL);
	if (!window) {
		glfwTerminate();
		throw std::runtime_error("Failed to create window");
	}
	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, mousePositionCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	// Set up GL context
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}
	printf("Opengl Version:%s\n", glGetString(GL_VERSION));
	//Set up ImGui

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	io = &ImGui::GetIO(); (void)io;
	const float fontSize = 22.0f;
	appFont = io->Fonts->AddFontFromFileTTF("..\\fonts\\cascadia-code\\Cascadia.ttf", fontSize);
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 120");

	

	// Initialize other stuff
	initVAO();
	initTextures();
	initCuda();
	initPBO();
	GLuint passthroughProgram = initShader();

	glUseProgram(passthroughProgram);
	glActiveTexture(GL_TEXTURE0);

	return true;
}

void InitImguiData(GuiDataContainer* guiData)
{
	imguiData = guiData;
}



// LOOK: Un-Comment to check ImGui Usage
void RenderImGui()
{
	mouseOverImGuiWinow = io->WantCaptureMouse;

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	bool show_demo_window = true;
	bool show_another_window = false;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
	static float f = 0.0f;
	static int counter = 0;

	ImGuiIO& io = ImGui::GetIO();
	
	ImGui::PushFont(appFont);

	ImGui::Begin("General");                  // Create a window called "Hello, world!" and append into it.
	
	// LOOK: Un-Comment to check the output window and usage
	//ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
	//ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
	//ImGui::Checkbox("Another Window", &show_another_window);

	//ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
	//ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

	//if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
	//	counter++;
	//ImGui::SameLine();
	//ImGui::Text("counter = %d", counter);
	ImGui::Text("Scene Path: %s", scene->sceneFilename.c_str());
	ImGui::Text("Traced Depth: %d", imguiData->TracedDepth);
	ImGui::Text("Integrator: %s", imguiData->integratorType.c_str());
	ImGui::Text("%.3f ms/iter (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	ImGui::Text("Elapsed time: %.2f sec", imguiData->elapsedTime);
	ImGui::Text("Num triangles: %d", scene->getTriangleSize());
	ImGui::Text("Num bvh nodes: %d", scene->bvhTreeSize);
	ImGui::Checkbox("Lock Controls", &lockControl);
	ImGui::End();

	ImGui::Begin("Camera Settings");
	ImGui::SliderFloat("Cam Speed", &camera_speed, 0.1f, 20.0f);

	if (ImGui::SliderFloat("Theta", &theta, 0.0f, PI))
		camchanged = true;
	if (ImGui::SliderFloat("Phi", &phi, 0.0f, TWO_PI))
		camchanged = true;
	if (ImGui::SliderFloat("Pos x", &renderState->camera.position.x, -100.0f, 100.0f))
		camchanged = true;
	if (ImGui::SliderFloat("Pos y", &renderState->camera.position.y, -100.0f, 100.0f))
		camchanged = true;
	if (ImGui::SliderFloat("Pos z", &renderState->camera.position.z, -100.0f, 100.0f))
		camchanged = true;
	if (ImGui::SliderFloat("Lens Radius", &renderState->camera.lensRadius, 0.0f, 0.15f))
		camchanged = true;
	if (ImGui::SliderFloat("Focal Length", &renderState->camera.focalLength, 0.1f, 20.0f))
		camchanged = true;

	ImGui::End();

	ImGui::PopFont();



	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

}

bool MouseOverImGuiWindow()
{
	return mouseOverImGuiWinow;
}

void mainLoop() {
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		uint64_t currTime = time(nullptr);
		delta_t = currTime - sysTime;
		sysTime = currTime;
		if (!lockControl)
		{
			if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			{
				camchanged = true;
				renderState->camera.position += renderState->camera.view * camera_speed;
			}
			if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			{
				camchanged = true;
				renderState->camera.position -= renderState->camera.view * camera_speed;
			}
			if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			{
				camchanged = true;
				renderState->camera.position -= renderState->camera.right * camera_speed;
			}
			if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			{
				camchanged = true;
				renderState->camera.position += renderState->camera.right * camera_speed;
			}
		}
			
		runCuda();
		

		string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
		glfwSetWindowTitle(window, title.c_str());
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glClear(GL_COLOR_BUFFER_BIT);

		// Binding GL_PIXEL_UNPACK_BUFFER back to default
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

		// Render ImGui Stuff
		RenderImGui();

		glfwSwapBuffers(window);
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(window);
	glfwTerminate();
}
