R"(
/* R for compiler includes: const char * test = #include file.cl; */

#define EPSILON 0.0000001
#define PI 3.14159265
#define BVH_STACK_SIZE 64

/**
 * @struct Ray
 * @brief Data type to store information about a ray
 */
typedef struct {
    float3 dir;
    float3 invDir;
    int3 rayDirSign;
} Ray;

/**
 * @struct TriangleIntersectionResult
 * @brief A struct to return the calculation results of a triangle intersection
 */
typedef struct {
    bool hit;
    uint pBestTriId;
    float3 pointHit;
    float hitDist;
} TriangleIntersectionResult;

/**
 * @brief Calculates the squared distance of two vectors
 * @param a First vector
 * @param b Second vector
 * @return The square distance
 */
float distanceSquare(float3 a, float3 b)
{
    float result = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
    return fabs(result);
}

/**
 * @brief Calculates whether a ray intersects a box
 * @param origin    The origin of the ray
 * @param ray       The ray
 * @param boxPtr    A pointer to the box data
 * @return          A boolean indicating whether the ray hits the box
 */
bool rayIntersectsBox(float3 origin, Ray ray, __global float2* boxPtr)
{
    float2 limitsX = boxPtr[0];
    float2 limitsY = boxPtr[1];
    float2 limitsZ = boxPtr[2];

    float limitsX2[2];
    float limitsY2[2];
    float limitsZ2[2];
    limitsX2[0] = limitsX.x;
    limitsX2[1] = limitsX.y;
    limitsY2[0] = limitsY.x;
    limitsY2[1] = limitsY.y;
    limitsZ2[0] = limitsZ.x;
    limitsZ2[1] = limitsZ.y;

    float tmin, tmax, tymin, tymax, tzmin, tzmax;

    tmin =  (limitsX2[    ray.rayDirSign.x] - origin.x) * ray.invDir.x;
    tmax =  (limitsX2[1 - ray.rayDirSign.x] - origin.x) * ray.invDir.x;
    tymin = (limitsY2[    ray.rayDirSign.y] - origin.y) * ray.invDir.y;
    tymax = (limitsY2[1 - ray.rayDirSign.y] - origin.y) * ray.invDir.y;

    if ((tmin > tymax) || (tymin > tmax))
    {
        return false;
    }
    if (tymin >tmin)
    {
        tmin = tymin;
    }
    if (tymax < tmax)
    {
        tmax = tymax;
    }

    tzmin = (limitsZ2[    ray.rayDirSign.z] - origin.z) * ray.invDir.z;
    tzmax = (limitsZ2[1 - ray.rayDirSign.z] - origin.z) * ray.invDir.z;

    if ((tmin > tzmax) || (tzmin > tmax))
    {
        return false;
    }
    if (tzmin > tmin)
    {
        tmin = tzmin;
    }
    if (tzmax < tmax)
    {
        tmax = tzmax;
    }

    return true;
}

/**
 * @brief Calculates the closest intersection of a raycast into a scene of triangles, given a bounding volume hierarchy
 *
 * @param clBVHindicesOrTriLists        Compressed BVH Node data, that stores for each node, whether it is a leaf node
 *                                      and triangle indices lists for leaf nodes and the indices of their child nodes
 *                                      for inner nodes
 * @param origin                        Origin of the ray
 * @param ray                           Direction of the ray
 * @param clBVHlimits                   3d upper and lower limits for each bounding box in the BVH
 * @param clTriangleIntersectionData    Precomputed intersection data for each triangle
 * @param clTriIdxList                  List of triangle indices
 * @return The TriangleIntersectionResult, containing information about the triangle intersection
 */
TriangleIntersectionResult intersectTrianglesBVH(
    __global uint* clBVHindicesOrTriLists,
    float3 origin,
    Ray ray,
    __global float2* clBVHlimits,
    __global float4* clTriangleIntersectionData,
    __global uint* clTriIdxList
)
{
    TriangleIntersectionResult result;
    result.hit = false;
    uint pBestTriId = 0;
    float bestTriDist = MAXFLOAT;

    uint stack[BVH_STACK_SIZE];

    int stackId = 0;
    stack[stackId++] = 0;
    float3 hitpoint;

    // while stack is not empty
    while (stackId)
    {
        uint boxId = stack[stackId - 1];

        stackId--;

        // the first bit of the data of a bvh node indicates whether it is a leaf node, by performing a bitwise and
        // with 0x80000000 all other bits are set to zero and the value of that one bit can be checked
        if (!(clBVHindicesOrTriLists[4 * boxId + 0] & 0x80000000)) // inner node
        {
            // if ray intersects inner node, push indices of left and right child nodes on the stack
            if (rayIntersectsBox(origin, ray, &clBVHlimits[3 * boxId]))
            {
                stack[stackId++] = clBVHindicesOrTriLists[4 * boxId + 1];
                stack[stackId++] = clBVHindicesOrTriLists[4 * boxId + 2];

                // return if stack size is exceeded
                if ( stackId > BVH_STACK_SIZE)
                {
                    printf("BVH stack size exceeded!");
                    result.hit = 0;
                    return result;
                }
            }
        }
        else // leaf node
        {
            // iterate over all triangles in this leaf node
            for (
                uint i = clBVHindicesOrTriLists[4 * boxId + 3];
                i < (clBVHindicesOrTriLists[4 * boxId + 3] + (clBVHindicesOrTriLists[4* boxId + 0] & 0x7fffffff));
                i++
            )
            {
                uint idx = clTriIdxList[i];
                float4 normal = clTriangleIntersectionData[4 * idx];

                float k = dot(normal.xyz, ray.dir);
                if (k == 0.0f)
                {
                    continue; // this triangle is parallel to the ray -> ignore it
                }
                float s = (normal.w - dot(normal.xyz, origin)) / k;
                if (s <= 0.0f)
                {
                    continue; // this triangle is "behind" the origin
                }
                if (s <= EPSILON)
                {
                    continue; // epsilon
                }
                float3 hit = ray.dir * s;
                hit += origin;

                // ray triangle intersection
                // check if the intersection with the triangle's plane is inside the triangle
                float4 ee1 = clTriangleIntersectionData[4 * idx + 1];
                float kt1 = dot(ee1.xyz, hit) - ee1.w;
                if (kt1 < 0.0f)
                {
                    continue;
                }
                float4 ee2 = clTriangleIntersectionData[4 * idx + 2];
                float kt2 = dot(ee2.xyz, hit) - ee2.w;
                if (kt2 < 0.0f)
                {
                    continue;
                }
                float4 ee3 = clTriangleIntersectionData[4 * idx + 3];
                float kt3 = dot(ee3.xyz, hit) - ee3.w;
                if (kt3 < 0.0f)
                {
                    continue;
                }

                // ray intersects triangle, "hit" is the coordinate of the intersection
                {
                    // check if this intersection closer than others
                    // use quadratic distance for comparison to save some root calculations
                    float hitZ = distanceSquare(origin, hit);
                    if (hitZ < bestTriDist)
                    {
                        bestTriDist = hitZ;
                        pBestTriId = idx;
                        result.hit = true;
                        hitpoint = hit;
                    }
                }

            }
        }
    }

    result.pBestTriId = pBestTriId ;
    result.pointHit = hitpoint;
    result.hitDist = sqrt(bestTriDist);

    return result;
}

float4 quat_from_axis_angle(float3 axis, float angle)
{
    float4 qr;

    float half_angle = (angle * 0.5) * PI / 180.0;
    qr.x = axis.x * sin(half_angle);
    qr.y = axis.y * sin(half_angle);
    qr.z = axis.z * sin(half_angle);
    qr.w = cos(half_angle);

    return qr;
}

float4 quat_conj(float4 q)
{
    return (float4)(-q.x, -q.y, -q.z, q.w);
}

float4 quat_mult(float4 q1, float4 q2)
{
    float4 qr;

    qr.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
    qr.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
    qr.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
    qr.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);

    return qr;
}

/**
 * @brief Rotates a vector around a given axis
 * @param position  The vector to be rotated
 * @param axis      The axis to rotate around
 * @param angle     The angle of the rotation
 * @return          The resulting vector
 */
float3 rotate_vertex_position(float3 position, float3 axis, float angle)
{
    //float4 qr = quat_from_axis_angle(axis, angle);
    //float4 qr_conj = quat_conj(qr);
    //float4 q_pos = (float4)(position.x, position.y, position.z, 0);
    //float4 q_tmp = quat_mult(qr, q_pos);
    //qr = quat_mult(q_tmp, qr_conj);
    //return (float3)(qr.x, qr.y, qr.z);

    // the above optimized
    float4 q = quat_from_axis_angle(axis, angle);
    return position + (2.0f * cross(q.xyz, cross(q.xyz, position) + q.w * position));
}

/**
 * @brief Casts rays into a scene of triangles, given a bounding volume hierarchy
 *
 * @param rays_origin                   Origin of the ray
 * @param rays                          Forward directions of the pose
 * @param clBVHindicesOrTriLists        Compressed BVH Node data, that stores for each node, whether it is a leaf node
 *                                      and triangle indices lists for leaf nodes and the indices of their child nodes
 *                                      for inner nodes
 * @param clBVHlimits                   3d upper and lower limits for each bounding box in the BVH
 * @param clTriangleIntersectionData    Precomputed intersection data for each triangle
 * @param clTriIdxList                  List of triangle indices
 * @param scannerFOV                    Field of view of the scanner model
 * @param result                        Result point positions
 * @param result_hits                   Result hits, where for each ray is stored whether it has hit a triangle
 */
__kernel void cast_rays(
    __global float* ray_origin,
    __global float* rays,
    __global uint* clBVHindicesOrTriLists,
    __global float2* clBVHlimits,
    __global float4* clTriangleIntersectionData,
    __global uint* clTriIdxList,
    __global float* scannerFOV,
    __global float* result,
    __global uchar* result_hits
)
{
    int idX = get_global_id(0);
    int idY = get_global_id(1);
    int numCols = get_global_size(0);
    int numRows = get_global_size(1);

    int id = (idY * numCols) + idX;

    float3 ray_d = (float3)(rays[0], rays[1], rays[2]);
    float3 ray_o = (float3)(ray_origin[0], ray_origin[1], ray_origin[2]);
    float horizontalFOV = scannerFOV[0];
    float verticalFOV = scannerFOV[1];

    // define y and x axis as approximation for up and side direction to rotate the ray
    float3 y_axis = (float3)(0, 0, 1);
    float3 x_axis = (float3)(0, 1, 0);

    // rotate the rays direction according to the current index
    ray_d = rotate_vertex_position(ray_d, x_axis, idY * (verticalFOV / (float)numRows) - (verticalFOV / 2.0));
    ray_d = rotate_vertex_position(ray_d, y_axis, idX * (horizontalFOV / (float)numCols) - (horizontalFOV / 2.0));

    // initialize result memory with zeros
    result[id*3] = 0;
    result[id*3+1] = 0;
    result[id*3+2] = 0;
    result_hits[id] = 0;


    // precompute ray values to speed up intersection calculation
    Ray ray;
    ray.dir = ray_d;
    ray.invDir = (float3)(1.0 / ray_d.x, 1.0 / ray_d.y, 1.0 / ray_d.z);
    ray.rayDirSign.x = ray.invDir.x < 0;
    ray.rayDirSign.y = ray.invDir.y < 0;
    ray.rayDirSign.z = ray.invDir.z < 0;


    // intersect all triangles stored in the BVH
    TriangleIntersectionResult resultBVH = intersectTrianglesBVH(
        clBVHindicesOrTriLists,
        ray_o,
        ray,
        clBVHlimits,
        clTriangleIntersectionData,
        clTriIdxList
    );

    // if a triangle was hit, store the calculated hit point in the result at the current id
    if (resultBVH.hit)
    {
        result[id*3] = resultBVH.pointHit.x;
        result[id*3 + 1] = resultBVH.pointHit.y;
        result[id*3 + 2] = resultBVH.pointHit.z;
        result_hits[id] = 1;
    }
}

/**
 * @brief Casts multiple rays with corresponding origins into a scene of triangles, given a bounding volume hierarchy
 *
 * @param rays_origin                   Origins of the rays
 * @param rays                          Forward directions of each pose
 * @param clBVHindicesOrTriLists        Compressed BVH Node data, that stores for each node, whether it is a leaf node
 *                                      and triangle indices lists for leaf nodes and the indices of their child nodes
 *                                      for inner nodes
 * @param clBVHlimits                   3d upper and lower limits for each bounding box in the BVH
 * @param clTriangleIntersectionData    Precomputed intersection data for each triangle
 * @param clTriIdxList                  List of triangle indices
 * @param scannerFOV                    Field of view of the scanner model
 * @param result                        Result point positions
 * @param result_hits                   Result hits, where for each ray is stored whether it has hit a triangle
 * @param ray_sides                     Side vectors for each pose
 * @param ray_ups                       Up vectors for each pose
 */
__kernel void cast_rays_multi_multi(
    __global float* ray_origin,
    __global float* rays,
    __global uint* clBVHindicesOrTriLists,
    __global float2* clBVHlimits,
    __global float4* clTriangleIntersectionData,
    __global uint* clTriIdxList,
    __global float* result,
    __global uchar* result_hits
)
{
    unsigned int id = get_global_id(0);

    // get direction and origin of the ray for the current pose
    float3 ray_d = (float3)(rays[id*3], rays[id*3+1], rays[id*3+2]);
    float3 ray_o = (float3)(ray_origin[id*3], ray_origin[id*3+1], ray_origin[id*3+2]);

    // initialize result memory with zeros
    result[id*3] = 0;
    result[id*3+1] = 0;
    result[id*3+2] = 0;
    result_hits[id] = 0;

    // precompute ray values to speed up intersection calculation
    Ray ray;
    ray.dir = ray_d;
    ray.invDir = (float3)(1.0 / ray_d.x, 1.0 / ray_d.y, 1.0 / ray_d.z);
    ray.rayDirSign.x = ray.invDir.x < 0;
    ray.rayDirSign.y = ray.invDir.y < 0;
    ray.rayDirSign.z = ray.invDir.z < 0;


    // intersect all triangles stored in the BVH
    TriangleIntersectionResult resultBVH = intersectTrianglesBVH(
        clBVHindicesOrTriLists,
        ray_o,
        ray,
        clBVHlimits,
        clTriangleIntersectionData,
        clTriIdxList
    );

    // if a triangle was hit, store the calculated hit point in the result at the current id
    if (resultBVH.hit)
    {
        result[id*3] = resultBVH.pointHit.x;
        result[id*3 + 1] = resultBVH.pointHit.y;
        result[id*3 + 2] = resultBVH.pointHit.z;
        result_hits[id] = 1;
    }
}

/**
 * @brief Casts multiple rays from one origin into a scene of triangles, given a bounding volume hierarchy
 *
 * @param rays_origin                   Origins of the rays
 * @param rays                          Forward directions of each pose
 * @param clBVHindicesOrTriLists        Compressed BVH Node data, that stores for each node, whether it is a leaf node
 *                                      and triangle indices lists for leaf nodes and the indices of their child nodes
 *                                      for inner nodes
 * @param clBVHlimits                   3d upper and lower limits for each bounding box in the BVH
 * @param clTriangleIntersectionData    Precomputed intersection data for each triangle
 * @param clTriIdxList                  List of triangle indices
 * @param scannerFOV                    Field of view of the scanner model
 * @param result                        Result point positions
 * @param result_hits                   Result hits, where for each ray is stored whether it has hit a triangle
 * @param ray_sides                     Side vectors for each pose
 * @param ray_ups                       Up vectors for each pose
 */
__kernel void cast_rays_one_multi(
    __global float* ray_origin,
    __global float* rays,
    __global uint* clBVHindicesOrTriLists,
    __global float2* clBVHlimits,
    __global float4* clTriangleIntersectionData,
    __global uint* clTriIdxList,
    __global float* result,
    __global uchar* result_hits
)
{
    unsigned int id = get_global_id(0);

    // get direction and origin of the ray for the current pose
    float3 ray_d = (float3)(rays[0], rays[1], rays[2]);
    float3 ray_o = (float3)(ray_origin[id*3], ray_origin[id*3+1], ray_origin[id*3+2]);

    // initialize result memory with zeros
    result[id*3] = 0;
    result[id*3+1] = 0;
    result[id*3+2] = 0;
    result_hits[id] = 0;

    // precompute ray values to speed up intersection calculation
    Ray ray;
    ray.dir = ray_d;
    ray.invDir = (float3)(1.0 / ray_d.x, 1.0 / ray_d.y, 1.0 / ray_d.z);
    ray.rayDirSign.x = ray.invDir.x < 0;
    ray.rayDirSign.y = ray.invDir.y < 0;
    ray.rayDirSign.z = ray.invDir.z < 0;


    // intersect all triangles stored in the BVH
    TriangleIntersectionResult resultBVH = intersectTrianglesBVH(
        clBVHindicesOrTriLists,
        ray_o,
        ray,
        clBVHlimits,
        clTriangleIntersectionData,
        clTriIdxList
    );

    // if a triangle was hit, store the calculated hit point in the result at the current id
    if (resultBVH.hit)
    {
        result[id*3] = resultBVH.pointHit.x;
        result[id*3 + 1] = resultBVH.pointHit.y;
        result[id*3 + 2] = resultBVH.pointHit.z;
        result_hits[id] = 1;
    }
}

/**
 * @brief Casts one ray from one origin into a scene of triangles, given a bounding volume hierarchy
 *
 * @param rays_origin                   Origins of the rays
 * @param rays                          Forward directions of each pose
 * @param clBVHindicesOrTriLists        Compressed BVH Node data, that stores for each node, whether it is a leaf node
 *                                      and triangle indices lists for leaf nodes and the indices of their child nodes
 *                                      for inner nodes
 * @param clBVHlimits                   3d upper and lower limits for each bounding box in the BVH
 * @param clTriangleIntersectionData    Precomputed intersection data for each triangle
 * @param clTriIdxList                  List of triangle indices
 * @param scannerFOV                    Field of view of the scanner model
 * @param result                        Result point positions
 * @param result_hits                   Result hits, where for each ray is stored whether it has hit a triangle
 * @param ray_sides                     Side vectors for each pose
 * @param ray_ups                       Up vectors for each pose
 */
__kernel void cast_rays_one_one(
    __global float* ray_origin,
    __global float* rays,
    __global uint* clBVHindicesOrTriLists,
    __global float2* clBVHlimits,
    __global float4* clTriangleIntersectionData,
    __global uint* clTriIdxList,
    __global float* result,
    __global uchar* result_hits
)
{
    // get direction and origin of the ray for the current pose
    float3 ray_d = (float3)(rays[0], rays[1], rays[2]);
    float3 ray_o = (float3)(ray_origin[0], ray_origin[1], ray_origin[2]);

    // initialize result memory with zeros
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    result_hits[0] = 0;

    // precompute ray values to speed up intersection calculation
    Ray ray;
    ray.dir = ray_d;
    ray.invDir = (float3)(1.0 / ray_d.x, 1.0 / ray_d.y, 1.0 / ray_d.z);
    ray.rayDirSign.x = ray.invDir.x < 0;
    ray.rayDirSign.y = ray.invDir.y < 0;
    ray.rayDirSign.z = ray.invDir.z < 0;


    // intersect all triangles stored in the BVH
    TriangleIntersectionResult resultBVH = intersectTrianglesBVH(
        clBVHindicesOrTriLists,
        ray_o,
        ray,
        clBVHlimits,
        clTriangleIntersectionData,
        clTriIdxList
    );

    // if a triangle was hit, store the calculated hit point in the result at the current id
    if (resultBVH.hit)
    {
        result[0] = resultBVH.pointHit.x;
        result[1] = resultBVH.pointHit.y;
        result[2] = resultBVH.pointHit.z;
        result_hits[0] = 1;
    }
}

__kernel void test(
    __global float* ray_origin,
    __global float* rays
)
{
    unsigned int id = get_global_id(0);
    printf("test kernel: %u\n", id);
    printf("number: %f\n", rays[0]);
}


)"