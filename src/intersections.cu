#include "interactions.h"

__device__ float util_geometry_ray_box_intersection(const glm::vec3& pMin, const glm::vec3& pMax, const Ray& r, bool* outside, glm::vec3* normal)
{
    float tmin = 0;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;

    for (int xyz = 0; xyz < 3; ++xyz) {
        float inv_qdxyz = 1.0f / r.direction[xyz];
        {
            float t1 = (pMin[xyz] - r.origin[xyz]) * inv_qdxyz;
            float t2 = (pMax[xyz] - r.origin[xyz]) * inv_qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }
    if (tmax >= tmin && tmax > 0) {
        *outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            *outside = false;
        }
        if (normal) *normal = tmin_n;
        return tmin;
    }
    return -1;
}

__device__ bool util_geometry_ray_box_intersection(const glm::vec3& pMin, const glm::vec3& pMax, const Ray& r, float t_max, float* t_min_p, float* t_max_p)
{
    float tmin = 0;
    float tmax = t_max;

    for (int xyz = 0; xyz < 3; ++xyz) {
        float inv_qdxyz = 1.0f / r.direction[xyz];
        {
            float t1 = (pMin[xyz] - r.origin[xyz]) * inv_qdxyz;
            float t2 = (pMax[xyz] - r.origin[xyz]) * inv_qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            if (ta > 0 && ta > tmin) {
                tmin = ta;
            }
            if (tb < tmax) {
                tmax = tb;
            }
        }
    }
    if (tmax >= tmin && tmax > 0) {
        *t_min_p = tmin;
        *t_max_p = tmax;
        return true;
    }
    return false;
}


__device__ float boxIntersectionTest(const Object& box, const Ray& r,
    glm::vec3& intersectionPoint, glm::vec3& normal) {
    Ray q;
    q.origin = multiplyMV(box.Transform.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(box.Transform.inverseTransform, glm::vec4(r.direction, 0.0f)));
    bool outside;
    glm::vec3 local_n;
    // This t is inside transformed box space
    float t = util_geometry_ray_box_intersection(glm::vec3(-0.5f), glm::vec3(0.5f), q, &outside, &local_n);

    if (t > 0) {
        intersectionPoint = multiplyMV(box.Transform.transform, glm::vec4(getPointOnRay(q, t), 1.0f));
        normal = glm::normalize(multiplyMV(box.Transform.invTranspose, glm::vec4(local_n, 0.0f)));
        // Get t in world space
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

__device__ float util_geometry_ray_sphere_intersection(const Object& sphere, const Ray& r,
    glm::vec3& intersectionPoint, glm::vec3& normal) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.Transform.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.Transform.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    }
    else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
    }
    else {
        t = max(t1, t2);
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.Transform.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.Transform.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

    return glm::length(r.origin - intersectionPoint);
}

__device__ bool util_geometry_ray_triangle_intersection(
    const glm::vec3& orig, const glm::vec3& dir,
    const glm::vec3& A, const glm::vec3& B, const glm::vec3& C,
    float& t, glm::vec3& normal, glm::vec3& baryCoord)
{
    glm::vec3 BA = B - A;
    glm::vec3 CA = C - A;
    glm::vec3 N = glm::cross(BA, CA); // N

    float NdotRayDir = glm::dot(N, dir);
    if (abs(NdotRayDir) < 1e-10f)
        return false;

    float d = glm::dot(-N, A);

    t = -(glm::dot(N, orig) + d) / NdotRayDir;

    if (t < 0) return false;
    glm::vec3 Q = orig + t * dir;

    glm::vec3 T;

    glm::vec3 edge0 = B - A;
    glm::vec3 vp0 = Q - A;
    T = glm::cross(edge0, vp0);
    if (glm::dot(N, T) < 0) return false;
    float areaC = glm::length(T);

    glm::vec3 edge1 = C - B;
    glm::vec3 vp1 = Q - B;
    T = glm::cross(edge1, vp1);
    if (glm::dot(N, T) < 0)  return false;
    float areaA = glm::length(T);

    glm::vec3 edge2 = A - C;
    glm::vec3 vp2 = Q - C;
    T = glm::cross(edge2, vp2);
    if (glm::dot(N, T) < 0) return false;
    float areaB = glm::length(T);

    float area = glm::length(N);
    N /= area;
    normal = N;
    baryCoord[0] = areaA / area;
    baryCoord[1] = areaB / area;
    baryCoord[2] = areaC / area;

    return true;
}

__device__ bool util_geometry_ray_triangle_intersection_watertight(
    const glm::vec3& ori, const glm::vec3& dir,
    const glm::vec3& p0, const glm::vec3& p1,
    const glm::vec3& p2, float tMax, float& t,
    glm::vec3& normal, glm::vec3& baryCoord)
{
    glm::vec3 N = glm::cross(p1 - p0, p2 - p0);
    float area = glm::length2(N);
    if (area == 0)
    {
        return false;
    }
    glm::vec3 p0t = p0 - ori;
    glm::vec3 p1t = p1 - ori;
    glm::vec3 p2t = p2 - ori;
    glm::vec3 abs_dir = glm::abs(dir);
    int kz = math::max_component_index(abs_dir);
    int kx = kz + 1; if (kx == 3) kx = 0;
    int ky = kx + 1; if (ky == 3) ky = 0;
    glm::vec3 d = math::permute(dir, { kx,ky,kz });
    p0t = math::permute(p0t, { kx,ky,kz });
    p1t = math::permute(p1t, { kx,ky,kz });
    p2t = math::permute(p2t, { kx,ky,kz });
    float Sx = -d.x / d.z;
    float Sy = -d.y / d.z;
    float Sz = 1 / d.z;
    
    p0t.x += Sx * p0t.z;
    p0t.y += Sy * p0t.z;
    p1t.x += Sx * p1t.z;
    p1t.y += Sy * p1t.z;
    p2t.x += Sx * p2t.z;
    p2t.y += Sy * p2t.z;

    float e0 = math::difference_of_products(p1t.x, p2t.y, p1t.y, p2t.x);
    float e1 = math::difference_of_products(p2t.x, p0t.y, p2t.y, p0t.x);
    float e2 = math::difference_of_products(p0t.x, p1t.y, p0t.y, p1t.x);


    if (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f)
    {
        double p2txp1ty = (double)p2t.x * (double)p1t.y;
        double p2typ1tx = (double)p2t.y * (double)p1t.x;
        e0 = (float)(p2typ1tx - p2txp1ty);
        double p0txp2ty = (double)p0t.x * (double)p2t.y;
        double p0typ2tx = (double)p0t.y * (double)p2t.x;
        e1 = (float)(p0typ2tx - p0txp2ty);
        double p1txp0ty = (double)p1t.x * (double)p0t.y;
        double p1typ0tx = (double)p1t.y * (double)p0t.x;
        e2 = (float)(p1typ0tx - p1txp0ty);
    }

    if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
        return false;
    float det = e0 + e1 + e2;
    if (det == 0.0f)
        return false;

    p0t.z *= Sz;
    p1t.z *= Sz;
    p2t.z *= Sz;
    float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
    if (det < 0 && (tScaled >= 0 || tScaled < tMax * det))
        return false;
    else if (det > 0 && (tScaled <= 0 || tScaled > tMax * det))
        return false;

    float invDet = 1 / det;
    float b0 = e0 * invDet, b1 = e1 * invDet, b2 = e2 * invDet;
    t = tScaled * invDet;
    assert(!math::is_nan(t));
#if 1
    float maxZt = math::max_component_value(abs(glm::vec3(p0t.z, p1t.z, p2t.z)));
    float deltaZ = math::gamma(3) * maxZt;

    float maxXt = math::max_component_value(abs(glm::vec3(p0t.x, p1t.x, p2t.x)));
    float maxYt = math::max_component_value(abs(glm::vec3(p0t.y, p1t.y, p2t.y)));
    float deltaX = math::gamma(5) * (maxXt + maxZt);
    float deltaY = math::gamma(5) * (maxYt + maxZt);

    float deltaE = 2 * (math::gamma(2) * maxXt * maxYt + deltaY * maxXt +
        deltaX * maxYt);

    float maxE = math::max_component_value(abs(glm::vec3(e0, e1, e2)));
    float deltaT = 3 * (math::gamma(3) * maxE * maxZt + deltaE * maxZt +
        deltaZ * maxE) * std::abs(invDet);
    if (t <= deltaT)
        return false;
#endif

    
    baryCoord = glm::vec3(b0, b1, b2);
    normal = N / sqrtf(math::max(area, 0.0f));
    return true;
}

__device__ float triangleIntersectionTest(const ObjectTransform& Transform, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const Ray& r, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec3& baryCoord)
{
    float t = -1.0;
    glm::vec3 v0w = multiplyMV(Transform.transform, glm::vec4(v0, 1.0f));
    glm::vec3 v1w = multiplyMV(Transform.transform, glm::vec4(v1, 1.0f));
    glm::vec3 v2w = multiplyMV(Transform.transform, glm::vec4(v2, 1.0f));
#if WATER_TIGHT_MESH_INTERSECTION
    if (util_geometry_ray_triangle_intersection_watertight(r.origin, r.direction, v0w, v1w, v2w, FLT_MAX, t, normal, baryCoord))
    {
        intersectionPoint = r.origin + r.direction * t;
        return t;
    }
#else
    if (util_geometry_ray_triangle_intersection(r.origin, r.direction, v0w, v1w, v2w, t, normal, baryCoord))
    {
        intersectionPoint = r.origin + r.direction * t;
        return t;
    }
#endif
    else
    {
        return -1;
    }
}

__device__ bool util_bvh_leaf_intersect(
    int primsStart,
    int primsEnd,
    const SceneInfoDev& dev_sceneInfo,
    Ray* ray,
    ShadeableIntersection* intersection
)
{
    glm::vec3 tmp_intersect, tmp_normal, tmp_baryCoord, tmp_tangent;
    float tmp_fsign;
    glm::vec2	tmp_uv;
    bool intersected = false;
    float t = -1.0;
    for (int i = primsStart; i != primsEnd; i++)
    {
        const Primitive& prim = dev_sceneInfo.dev_primitives[i];
        int objID = prim.objID;
        const Object& obj = dev_sceneInfo.dev_objs[objID];

        if (obj.type == TRIANGLE_MESH)
        {
            const glm::ivec3& tri = dev_sceneInfo.modelInfo.dev_triangles[obj.triangleStart + prim.offset];
            const glm::vec3& v0 = dev_sceneInfo.modelInfo.dev_vertices[tri[0]];
            const glm::vec3& v1 = dev_sceneInfo.modelInfo.dev_vertices[tri[1]];
            const glm::vec3& v2 = dev_sceneInfo.modelInfo.dev_vertices[tri[2]];
            t = triangleIntersectionTest(obj.Transform, v0, v1, v2, *ray, tmp_intersect, tmp_normal, tmp_baryCoord);
            if (dev_sceneInfo.modelInfo.dev_uvs)
            {
                const glm::vec2& uv0 = dev_sceneInfo.modelInfo.dev_uvs[tri[0]];
                const glm::vec2& uv1 = dev_sceneInfo.modelInfo.dev_uvs[tri[1]];
                const glm::vec2& uv2 = dev_sceneInfo.modelInfo.dev_uvs[tri[2]];
                tmp_uv = uv0 * tmp_baryCoord[0] + uv1 * tmp_baryCoord[1] + uv2 * tmp_baryCoord[2];
            }
            if (dev_sceneInfo.modelInfo.dev_normals)
            {
                const glm::vec3& n0 = dev_sceneInfo.modelInfo.dev_normals[tri[0]];
                const glm::vec3& n1 = dev_sceneInfo.modelInfo.dev_normals[tri[1]];
                const glm::vec3& n2 = dev_sceneInfo.modelInfo.dev_normals[tri[2]];
                tmp_normal = n0 * tmp_baryCoord[0] + n1 * tmp_baryCoord[1] + n2 * tmp_baryCoord[2];
                tmp_normal = glm::vec3(obj.Transform.invTranspose * glm::vec4(tmp_normal, 0.0));//TODO: precompute transformation
            }
            if (dev_sceneInfo.modelInfo.dev_tangents)
            {
                const glm::vec3& t0 = dev_sceneInfo.modelInfo.dev_tangents[tri[0]];
                const glm::vec3& t1 = dev_sceneInfo.modelInfo.dev_tangents[tri[1]];
                const glm::vec3& t2 = dev_sceneInfo.modelInfo.dev_tangents[tri[2]];
                tmp_tangent = t0 * tmp_baryCoord[0] + t1 * tmp_baryCoord[1] + t2 * tmp_baryCoord[2];
                tmp_tangent = glm::vec3(obj.Transform.invTranspose * glm::vec4(tmp_tangent, 0.0));
            }
        }
        else if (obj.type == CUBE)
        {
            t = boxIntersectionTest(obj, *ray, tmp_intersect, tmp_normal);
        }
        else if (obj.type == SPHERE)
        {
            t = util_geometry_ray_sphere_intersection(obj, *ray, tmp_intersect, tmp_normal);
        }

        if (t > 0.0 && t < intersection->t)
        {
            intersection->t = t;
            intersection->materialId = obj.materialid;
            intersection->worldPos = tmp_intersect;
            intersection->surfaceNormal = tmp_normal;
            intersection->surfaceTangent = tmp_tangent;
            intersection->fsign = tmp_fsign;
            intersection->uv = tmp_uv;
            intersection->primitiveId = i;
            intersected = true;
        }

    }
    const Primitive& prim = dev_sceneInfo.dev_primitives[intersection->primitiveId];
    int objID = prim.objID;
    const Object& obj = dev_sceneInfo.dev_objs[objID];
    if (obj.mediumIn != obj.mediumOut)
    {
        ray->medium = glm::dot(ray->direction, intersection->surfaceNormal) > 0 ? obj.mediumIn : obj.mediumOut;
    }
    return intersected;
}

__device__ inline float util_bvh_leaf_test_intersect(
    int primsStart,
    int primsEnd,
    const SceneInfoDev& dev_sceneInfo,
    const Ray& ray
)
{
    glm::vec3 tmp_intersect, tmp_normal, tmp_baryCoord;
    float t, tmin = 1e37f;
    for (int i = primsStart; i != primsEnd; i++)
    {
        const Primitive& prim = dev_sceneInfo.dev_primitives[i];
        int objID = prim.objID;
        const Object& obj = dev_sceneInfo.dev_objs[objID];
        if (obj.type == TRIANGLE_MESH)
        {
            const glm::ivec3& tri = dev_sceneInfo.modelInfo.dev_triangles[obj.triangleStart + prim.offset];
            const glm::vec3& v0 = dev_sceneInfo.modelInfo.dev_vertices[tri[0]];
            const glm::vec3& v1 = dev_sceneInfo.modelInfo.dev_vertices[tri[1]];
            const glm::vec3& v2 = dev_sceneInfo.modelInfo.dev_vertices[tri[2]];
            t = triangleIntersectionTest(obj.Transform, v0, v1, v2, ray, tmp_intersect, tmp_normal, tmp_baryCoord);
        }
        else if (obj.type == CUBE)
        {
            t = boxIntersectionTest(obj, ray, tmp_intersect, tmp_normal);
        }
        else if (obj.type == SPHERE)
        {
            t = util_geometry_ray_sphere_intersection(obj, ray, tmp_intersect, tmp_normal);
        }
        if (t > 0.0 && t < tmin)
        {
            tmin = t;
        }
    }
    return tmin;
}

__device__ bool util_test_visibility(glm::vec3 p0, glm::vec3 p1, const SceneInfoDev& dev_sceneInfo)
{
    glm::vec3 dir = p1 - p0;
    if (glm::length(dir) < 0.001f) return true;
    Ray ray;
    ray.direction = glm::normalize(dir);
    ray.origin = p0;
    glm::vec3 t3 = (dir / ray.direction);
    float t, tmax = max(t3.x, max(t3.y, t3.z)) - 0.001f;
    glm::vec3 tmp_intersect, tmp_normal, tmp_baryCoord;
    for (int i = 0; i < dev_sceneInfo.objectsSize; i++)
    {
        Object& obj = dev_sceneInfo.dev_objs[i];
        if (obj.type == GeomType::CUBE)
        {
            t = boxIntersectionTest(obj, ray, tmp_intersect, tmp_normal);
            if (t > 0.0 && t < tmax) return false;
        }
        else if (obj.type == GeomType::SPHERE)
        {
            t = util_geometry_ray_sphere_intersection(obj, ray, tmp_intersect, tmp_normal);
            if (t > 0.0 && t < tmax) return false;
        }
        else
        {
            for (int j = obj.triangleStart; j != obj.triangleEnd; j++)
            {
                const glm::ivec3& tri = dev_sceneInfo.modelInfo.dev_triangles[j];
                const glm::vec3& v0 = dev_sceneInfo.modelInfo.dev_vertices[tri[0]];
                const glm::vec3& v1 = dev_sceneInfo.modelInfo.dev_vertices[tri[1]];
                const glm::vec3& v2 = dev_sceneInfo.modelInfo.dev_vertices[tri[2]];
                t = triangleIntersectionTest(obj.Transform, v0, v1, v2, ray, tmp_intersect, tmp_normal, tmp_baryCoord);
                if (t > 0.0 && t < tmax) return false;
            }
        }

    }
    return true;
}

__device__ bool util_bvh_test_visibility(glm::vec3 p0, glm::vec3 p1, const SceneInfoDev& dev_sceneInfo)
{
    glm::vec3 dir = p1 - p0;
    if (glm::length(dir) < 0.001f) return true;
    Ray ray;
    ray.direction = glm::normalize(dir);
    ray.origin = p0;
    glm::vec3 t3 = (dir / ray.direction);
    float tmax = max(t3.x, max(t3.y, t3.z)) - 0.001f;
#if MTBVH
    float x = fabs(ray.direction.x), y = fabs(ray.direction.y), z = fabs(ray.direction.z);
    int axis = x > y && x > z ? 0 : (y > z ? 1 : 2);
    int sgn = ray.direction[axis] > 0 ? 0 : 1;
    int d = (axis << 1) + sgn;
    const MTBVHGPUNode* currArray = dev_sceneInfo.dev_mtbvhArray + d * dev_sceneInfo.bvhDataSize;
    int curr = 0;

    while (curr >= 0 && curr < dev_sceneInfo.bvhDataSize)
    {
        bool outside = true;
        float boxt = boundingBoxIntersectionTest(currArray[curr].bbox, ray, outside);
        if (!outside) boxt = EPSILON;
        if (boxt > 0 && boxt < tmax + EPSILON)
        {
            if (currArray[curr].startPrim != -1)//leaf node
            {
                int start = currArray[curr].startPrim, end = currArray[curr].endPrim;
                if (util_bvh_leaf_test_intersect(start, end, dev_sceneInfo, ray) < tmax - EPSILON)
                    return false;
            }
            curr = currArray[curr].hitLink;
        }
        else
        {
            curr = currArray[curr].missLink;
        }
    }
#else
#endif // MTBVH
    return true;
}