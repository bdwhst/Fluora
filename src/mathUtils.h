#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <optional>
#include <glm/glm.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

// single precision floating point math library for ray tracing
namespace math
{
    constexpr float pi = 3.141592653589793f;
    constexpr float two_pi = pi * 2.0f;
    __device__ __host__ inline bool is_nan(float x)
    {
        return x != x;
    }
    template<typename T, typename S>
    __device__ __host__ inline T max(T x, S y)
    {
        return x > y ? x : (T)y;
    }

    template<typename T, typename S>
    __device__ __host__ inline T min(T x, S y)
    {
        return y < x ? (T)y : x;
    }

    template<typename T, typename S>
    __device__ __host__ inline S lerp(T x, S a, S b)
    {
        return (T(1.0f) - x) * a + x * b;
    }
    template<typename T>
    __device__ __host__ inline float sqr(T x){ return x*x; }

    template<typename T>
    __device__ __host__ inline float abs(T x) { return std::abs(x); }

    template <typename T, typename U, typename V>
    __device__ __host__ inline constexpr T clamp(T val, U low, V high) {
    if (val < low)
        return T(low);
    else if (val > high)
        return T(high);
    else
        return val;
    }
    //find position i such that f returns true for all position less or equal to i
    //such that f returns false for all position greater than i
    template <typename F>
    __device__ __host__ inline size_t find_interval(size_t sz, const F& f)
    {
        using ssize_t = std::make_signed_t<size_t>;
        ssize_t size = (ssize_t)sz - 2, first = 1;
        while (size > 0)
        {
            size_t half = (size_t)size >> 1;
            size_t mid = first + half;
            bool p = f(mid);
            first = p ? mid + 1 : first;
            size = p ? size - half - 1 : half;
        }
        return (size_t)clamp((ssize_t)first - 1, 0, sz - 2);
    }

    /*__device__ __host__ inline float FMA(float a, float b, float c) {
        return fma(a, b, c);
    }

    __device__ __host__ inline double FMA(double a, double b, double c) {
        return fma(a, b, c);
    }*/

    template <typename Float, typename C>
    __device__ __host__  inline constexpr Float evaluate_polynomial(Float t, C c) {
        return c;
    }


    template <typename Float, typename C, typename... Args>
    __device__ __host__ inline constexpr Float evaluate_polynomial(Float t, C c, Args... cRemaining) {
        return fma(t, evaluate_polynomial(t, cRemaining...), c);
    }

    // https://github.com/explosion/murmurhash/blob/master/murmurhash/MurmurHash2.cpp
    __device__ __host__  inline uint64_t MurmurHash64A(const unsigned char* key, size_t len,
        uint64_t seed) {
        const uint64_t m = 0xc6a4a7935bd1e995ull;
        const int r = 47;

        uint64_t h = seed ^ (len * m);

        const unsigned char* end = key + 8 * (len / 8);

        while (key != end) {
            uint64_t k;
            std::memcpy(&k, key, sizeof(uint64_t));
            key += 8;

            k *= m;
            k ^= k >> r;
            k *= m;

            h ^= k;
            h *= m;
        }

        switch (len & 7) {
        case 7:
            h ^= uint64_t(key[6]) << 48;
        case 6:
            h ^= uint64_t(key[5]) << 40;
        case 5:
            h ^= uint64_t(key[4]) << 32;
        case 4:
            h ^= uint64_t(key[3]) << 24;
        case 3:
            h ^= uint64_t(key[2]) << 16;
        case 2:
            h ^= uint64_t(key[1]) << 8;
        case 1:
            h ^= uint64_t(key[0]);
            h *= m;
        };

        h ^= h >> r;
        h *= m;
        h ^= h >> r;

        return h;
    }

    template <typename T>
    __device__ __host__ inline uint64_t HashBuffer(const T* ptr, size_t size, uint64_t seed = 0) {
        return MurmurHash64A((const unsigned char*)ptr, size, seed);
    }


    template <int N>
    class SquareMatrix {
    public:
        __device__ __host__ static SquareMatrix zero()
        {
            SquareMatrix mat;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    mat.m[i][j] = 0;
            return mat;

        }
        __device__ __host__ SquareMatrix()
        {
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    m[i][j] = (i == j) ? 1 : 0;
        }

        __device__ __host__ SquareMatrix(float v)
        {
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    m[i][j] = v;
        }
        __device__ __host__
            SquareMatrix(const float mat[N][N]) {
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    m[i][j] = mat[i][j];
        }

        __device__ __host__
            SquareMatrix operator+(const SquareMatrix& m) const {
            SquareMatrix r = *this;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    r.m[i][j] += m.m[i][j];
            return r;
        }

        __device__ __host__
            SquareMatrix operator*(float s) const {
            SquareMatrix r = *this;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    r.m[i][j] *= s;
            return r;
        }
        __device__ __host__
            SquareMatrix operator/(float s) const {
            SquareMatrix r = *this;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    r.m[i][j] /= s;
            return r;
        }

        __device__ __host__
            bool operator==(const SquareMatrix<N>& m2) const {
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    if (m[i][j] != m2.m[i][j])
                        return false;
            return true;
        }

        __device__ __host__
            bool operator!=(const SquareMatrix<N>& m2) const {
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j)
                    if (m[i][j] != m2.m[i][j])
                        return true;
            return false;
        }

        __device__ __host__
            float* operator[](int i) const { return (float*)m[i]; }

        float m[N][N];
    };

    template <int N>
    __device__ __host__ inline SquareMatrix<N> operator*(const SquareMatrix<N>& m1,
        const SquareMatrix<N>& m2) {
        SquareMatrix<N> r;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                r[i][j] = 0;
                for (int k = 0; k < N; ++k)
                    r[i][j] = std::fma(m1[i][k], m2[k][j], r[i][j]);
            }
        return r;
    }

    template <int N>
    __device__ __host__ inline SquareMatrix<N> transpose(const SquareMatrix<N>& m) {
        SquareMatrix<N> r;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                r[i][j] = m[j][i];
        return r;
    }

    template <int N>
    __device__ __host__ bool inverse(const SquareMatrix<N>& m, SquareMatrix<N>& out) {
        int indxc[N], indxr[N];
        int ipiv[N] = { 0 };
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                out[i][j] = m[i][j];
        for (int i = 0; i < N; i++) {
            int irow = 0, icol = 0;
            float big = 0.f;
            // Choose pivot
            for (int j = 0; j < N; j++) {
                if (ipiv[j] != 1) {
                    for (int k = 0; k < N; k++) {
                        if (ipiv[k] == 0) {
                            if (std::abs(out[j][k]) >= big) {
                                big = std::abs(out[j][k]);
                                irow = j;
                                icol = k;
                            }
                        }
                        else if (ipiv[k] > 1)
                            return false;  // singular
                    }
                }
            }
            ++ipiv[icol];
            // Swap rows _irow_ and _icol_ for pivot
            if (irow != icol) {
                for (int k = 0; k < N; ++k)
                    std::swap(out[irow][k], out[icol][k]);
            }
            indxr[i] = irow;
            indxc[i] = icol;
            if (out[icol][icol] == 0.f)
                return false;  // singular

            // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
            float pivinv = 1. / out[icol][icol];
            out[icol][icol] = 1.;
            for (int j = 0; j < N; j++)
                out[icol][j] *= pivinv;

            // Subtract this row from others to zero out their columns
            for (int j = 0; j < N; j++) {
                if (j != icol) {
                    float save = out[j][icol];
                    out[j][icol] = 0;
                    for (int k = 0; k < N; k++)
                        out[j][k] = std::fma(-out[icol][k], save, out[j][k]);
                }
            }
        }
        // Swap columns to reflect permutation
        for (int j = N - 1; j >= 0; j--) {
            if (indxr[j] != indxc[j]) {
                for (int k = 0; k < N; k++)
                    std::swap(out[k][indxr[j]], out[k][indxc[j]]);
            }
        }
        return true;
    }

    template <int N>
    __device__ __host__
    bool linear_least_squares(const float A[][N], const float B[][N], int rows, SquareMatrix<N>& out) {
        SquareMatrix<N> AtA(0.0f);
        SquareMatrix<N> AtB(0.0f);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int r = 0; r < rows; ++r) {
                    AtA[i][j] += A[r][i] * A[r][j];
                    AtB[i][j] += A[r][i] * B[r][j];
                }
            }
        }
        SquareMatrix<N> AtAi;
        if (!inverse(AtA, AtAi)) {
            return false;
        }
        out = transpose(AtAi * AtB);
        return true;
    }

    __device__ inline float l2norm_squared(const glm::vec3& v)
    {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }
    __device__ inline float l2norm_squared(const glm::vec2& v)
    {
        return v.x * v.x + v.y * v.y;
    }
    __device__ inline float dist2(const glm::vec3& v0, const glm::vec3& v1)
    {
        return l2norm_squared(v0 - v1);
    }
    __device__ inline float sin_cos_convert(float sincos)
    {
        return sqrtf(max(0.0f, 1.0f - sqr(sincos)));
    }

    __device__ inline float cos_theta_vec(const glm::vec3& v)
    {
        return v.z;
    }
    __device__ inline float cos2_theta_vec(const glm::vec3& v)
    {
        return v.z * v.z;
    }
    __device__ inline float sin2_theta_vec(const glm::vec3& v)
    {
        return max(0.0f, 1.0f - cos2_theta_vec(v));
    }
    __device__ inline float sin_theta_vec(const glm::vec3& v)
    {
        return sqrtf(sin2_theta_vec(v));
    }
    __device__ inline float tan2_theta_vec(const glm::vec3& v)
    {
        return sin2_theta_vec(v) / cos2_theta_vec(v);
    }
    __device__ inline float cos_phi_vec(const glm::vec3& v)
    {
        float sintheta = sin_theta_vec(v);
        return sintheta == 0 ? 1 : clamp(v.x / sintheta, -1.0f, 1.0f);
    }
    __device__ inline float sin_phi_vec(const glm::vec3& v)
    {
        float sintheta = sin_theta_vec(v);
        return sintheta == 0 ? 1 : clamp(v.y / sintheta, -1.0f, 1.0f);
    }

    template<typename T>
    __device__ inline bool is_inf(T x)
    {
        return isinf(x);
    }

    __device__ inline float abs_dot(const glm::vec3& w0, const glm::vec3& w1)
    {
        return abs(glm::dot(w0, w1));
    }


    template <typename T>
    struct complex {
        __device__ __host__ complex(T re) : re(re), im(0) {}
        __device__ __host__  complex(T re, T im) : re(re), im(im) {}

        __device__ __host__  complex operator-() const { return { -re, -im }; }

        __device__ __host__  complex operator+(complex z) const { return { re + z.re, im + z.im }; }

        __device__ __host__  complex operator-(complex z) const { return { re - z.re, im - z.im }; }

        __device__ __host__  complex operator*(complex z) const {
            return { re * z.re - im * z.im, re * z.im + im * z.re };
        }

        __device__ __host__  complex operator/(complex z) const {
            T scale = 1 / (z.re * z.re + z.im * z.im);
            return { scale * (re * z.re + im * z.im), scale * (im * z.re - re * z.im) };
        }

        friend __device__ __host__  complex operator+(T value, complex z) {
            return complex(value) + z;
        }

        friend __device__ __host__  complex operator-(T value, complex z) {
            return complex(value) - z;
        }

        friend __device__ __host__  complex operator*(T value, complex z) {
            return complex(value) * z;
        }

        friend __device__ __host__  complex operator/(T value, complex z) {
            return complex(value) / z;
        }

        T re, im;
    };

    template<typename T>
    __device__ __host__ complex<T> sqr(const complex<T>& z) { return z * z; }

    template <typename T>
    __device__ __host__  T norm(const complex<T>& z) {
        return z.re * z.re + z.im * z.im;
    }

    template <typename T>
    __device__ __host__  T abs(const complex<T>& z) {
        return sqrtf(norm(z));
    }

    template <typename T>
    __device__ __host__ float safe_sqrt(T x)
    {
        return sqrtf(max(0.0f, (float)x));
    }

    template <typename T>
    __device__ __host__ complex<T> sqrt(const complex<T>& z) {
        T n = abs(z), t1 = sqrtf(T(.5) * (n + abs(z.re))),
            t2 = T(.5) * z.im / t1;

        if (n == 0)
            return 0;

        if (z.re >= 0)
            return { t1, t2 };
        else
            return { abs(t2), copysignf(t1, z.im) };
    }
    __device__ glm::vec2 sample_uniform_disk_polar(const glm::vec2& u);
    __device__ inline glm::vec3 reflect(const glm::vec3& wo, const glm::vec3& n)
    {
        return -wo + 2 * glm::dot(wo, n) * n;
    }
    __device__ inline bool sample_hemisphere(const glm::vec3& w1, const glm::vec3& w2)
    {
        return w1.z * w2.z > 0;
    }

    __device__ inline glm::vec2 sample_disk_uniform(const glm::vec2& random)
    {
        float r = sqrtf(random.x);
        float theta = two_pi * random.y;
        return glm::vec2(r * cos(theta), r * sin(theta));
    }

    __device__ inline glm::vec3 sample_hemisphere_cosine(const glm::vec2& random)
    {
        glm::vec2 t = sample_disk_uniform(random);
        return glm::vec3(t.x, t.y, sqrtf(1 - t.x * t.x - t.y * t.y));
    }

    __device__ inline float frensel_dielectric(float cosThetaI, float etaI, float etaT)
    {
        float sinThetaI = sin_cos_convert(cosThetaI);
        float sinThetaT = etaI / etaT * sinThetaI;
        if (sinThetaT >= 1) return 1;//total reflection
        float cosThetaT = sin_cos_convert(sinThetaT);
        float rparll = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
        float rperpe = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
        return (rparll * rparll + rperpe * rperpe) * 0.5;
    }

    __device__ inline bool geomerty_refract(const glm::vec3& wi, const glm::vec3& n, float eta, glm::vec3* wt)
    {
        float cosThetaI = glm::dot(wi, n);
        float sin2ThetaI = max(0.0f, 1 - cosThetaI * cosThetaI);
        float sin2ThetaT = eta * eta * sin2ThetaI;
        if (sin2ThetaT >= 1) return false;
        float cosThetaT = sqrtf(1 - sin2ThetaT);
        *wt = eta * (-wi) + (eta * cosThetaI - cosThetaT) * n;
        return true;
    }

    __device__ inline glm::vec3 spherical_direction(float sinTheta, float cosTheta, float phi)
    {
        sinTheta = clamp(sinTheta, -1, 1);
        cosTheta = clamp(cosTheta, -1, 1);
        return glm::vec3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
    }


    //http://marc-b-reynolds.github.io/quaternions/2016/07/06/Orthonormal.html
    //Get tangent space vectors
    __device__ void get_tbn_pixar(const glm::vec3& N, glm::vec3* T, glm::vec3* B);

    class Frame 
    {
        glm::vec3 x, y, z;
    public:
        __device__ Frame(const glm::vec3& x, const glm::vec3& y, const glm::vec3& z):x(x),y(y),z(z){}
        __device__ static Frame from_z(const glm::vec3& z)
        {
            glm::vec3 x, y;
            get_tbn_pixar(z, &x, &y);
            return Frame(x, y, z);
        }
        __device__ glm::vec3 to_local(const glm::vec3& v)
        {
            return glm::vec3(glm::dot(v, x), glm::dot(v, y), glm::dot(v, z));
        }
        __device__ glm::vec3 from_local(const glm::vec3& v)
        {
            return v.x * x + v.y * y + v.z * z;
        }
    };

    __device__ inline float sample_exponential(float u, float a)
    {
        assert(a > 0);
        return -logf(max(1.0f - u, std::numeric_limits<float>::denorm_min())) / a;
    }

    __device__ inline glm::vec3 permute(const glm::vec3& input, const glm::ivec3& idx)
    {
        assert(idx.x >= 0 && idx.x < 3 && idx.y >= 0 && idx.y < 3 && idx.z >= 0 && idx.z < 3);
        return { input[idx[0]], input[idx[1]], input[idx[2]] };
    }

    template <typename Ta, typename Tb, typename Tc, typename Td>
    __device__ inline auto difference_of_products(Ta a, Tb b, Tc c, Td d)
    {
        auto cd = c * d;
        auto ans = fma(a, b, -cd);
        auto err = fma(-c, d, cd);
        return ans + err;
    }

    __device__ inline int max_component_index(const glm::vec3& vec)
    {
        return (vec.x > vec.y && vec.x > vec.z) ? 0 : (vec.y > vec.z ? 1 : 2);
    }

    __device__ inline int max_component_value(const glm::vec3& vec)
    {
        return (vec.x > vec.y && vec.x > vec.z) ? vec.x : (vec.y > vec.z ? vec.y : vec.z);
    }

    constexpr float MachineEpsilon = std::numeric_limits<float>::epsilon() * 0.5;

    constexpr float gamma(int n)
    {
        return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
    }

    // Would be nice to allow Float to be a template type here, but it is tricky:
    // https://stackoverflow.com/questions/5101516/why-function-template-cannot-be-partially-specialized
    template <int n>
    __device__ inline constexpr float pow(float v) {
        if constexpr (n < 0)
            return 1 / pow<-n>(v);
        float n2 = pow<n / 2>(v);
        return n2 * n2 * pow<n & 1>(v);
    }

    template <>
    __device__ inline constexpr float pow<1>(float v) {
        return v;
    }
    template <>
    __device__ inline constexpr float pow<0>(float v) {
        return 1;
    }
};

