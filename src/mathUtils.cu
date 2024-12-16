#include "mathUtils.h"

namespace math
{
    __device__ glm::vec2 sample_uniform_disk_polar(const glm::vec2& u)
    {
        float r = sqrtf(u.x);
        float theta = 2 * pi * u[1];
        return { r * cosf(theta),r * sinf(theta) };
    }

    __device__ void get_tbn_pixar(const glm::vec3& N, glm::vec3* T, glm::vec3* B)
    {
        float x = N.x, y = N.y, z = N.z;
        float sz = z < 0 ? -1 : 1;
        float a = 1.0f / (sz + z);
        float ya = y * a;
        float b = x * ya;
        float c = x * sz;
        (*T) = glm::vec3(c * x * a - 1, sz * b, c);
        (*B) = glm::vec3(b, y * ya - sz, y);
    }

    
}