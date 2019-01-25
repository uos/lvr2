#pragma once

#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/geometry/Point.hpp>
#include <lvr2/geometry/BVH.hpp>
#include <lvr2/algorithm/raycasting/RaycasterBase.hpp>

#define EPSILON 0.0000001
#define PI 3.14159265
#define BVH_STACK_SIZE 64

namespace lvr2
{



/**
 *  @brief BVHRaycaster: CPU version of BVH Raycasting: WIP
 */
template <typename BaseVecT>
class BVHRaycaster : public RaycasterBase<BaseVecT> {
public:

    /**
     * @brief Constructor: Stores mesh as member
     */
    BVHRaycaster(const MeshBufferPtr mesh);

    bool castRay(
        const Point<BaseVecT>& origin,
        const Vector<BaseVecT>& direction,
        Point<BaseVecT>& intersection
    );

    void castRays(
        const Point<BaseVecT>& origin,
        const std::vector<Vector<BaseVecT> >& directions,
        std::vector<Point<BaseVecT> >& intersections,
        std::vector<uint8_t>& hits
    );

    void castRays(
        const std::vector<Point<BaseVecT> >& origins,
        const std::vector<Vector<BaseVecT> >& directions,
        std::vector<Point<BaseVecT> >& intersections,
        std::vector<uint8_t>& hits
    );


    /**
     * @struct Ray
     * @brief Data type to store information about a ray
     */

    struct Ray {
        Vector<BaseVecT> dir;
        Vector<BaseVecT> invDir;
        Vector<BaseVector<int> > rayDirSign;
    };


    /**
     * @struct TriangleIntersectionResult
     * @brief A struct to return the calculation results of a triangle intersection
     */
    struct TriangleIntersectionResult {
        bool hit;
        unsigned int pBestTriId;
        Vector<BaseVecT> pointHit;
        float hitDist;
    };
    

protected:
    BVHTree<BaseVecT> m_bvh;

private:


    /**
     * @brief Calculates the squared distance of two vectors
     * @param a First vector
     * @param b Second vector
     * @return The square distance
     */
    inline float distanceSquare(Vector<BaseVecT> a, Vector<BaseVecT> b)
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
    bool rayIntersectsBox(Vector<BaseVecT> origin, Ray ray, const float* boxPtr);

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
        const unsigned int* clBVHindicesOrTriLists,
        Vector<BaseVecT> origin,
        Ray ray,
        const float* clBVHlimits,
        const float* clTriangleIntersectionData,
        const unsigned int* clTriIdxList
    );

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
     * @param result                        Result point positions
     * @param result_hits                   Result hits, where for each ray is stored whether it has hit a triangle
     */
    void cast_rays_one_one(
        const float* ray_origin,
        const float* rays,
        const unsigned int* clBVHindicesOrTriLists,
        const float* clBVHlimits,
        const float* clTriangleIntersectionData,
        const unsigned int* clTriIdxList,
        float* result,
        uint8_t* result_hits
    );

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
     * @param result                        Result point positions
     * @param result_hits                   Result hits, where for each ray is stored whether it has hit a triangle
     */

    void cast_rays_one_multi(
        const float* ray_origin,
        const float* rays,
        size_t num_rays,
        const unsigned int* clBVHindicesOrTriLists,
        const float* clBVHlimits,
        const float* clTriangleIntersectionData,
        const unsigned int* clTriIdxList,
        float* result,
        uint8_t* result_hits
    );

    /**
     * @brief Casts multiple rays from multiple origins into a scene of triangles, given a bounding volume hierarchy
     *
     * @param rays_origin                   Origins of the rays
     * @param rays                          Forward directions of each pose
     * @param clBVHindicesOrTriLists        Compressed BVH Node data, that stores for each node, whether it is a leaf node
     *                                      and triangle indices lists for leaf nodes and the indices of their child nodes
     *                                      for inner nodes
     * @param clBVHlimits                   3d upper and lower limits for each bounding box in the BVH
     * @param clTriangleIntersectionData    Precomputed intersection data for each triangle
     * @param clTriIdxList                  List of triangle indices
     * @param result                        Result point positions
     * @param result_hits                   Result hits, where for each ray is stored whether it has hit a triangle
     */
    void cast_rays_multi_multi(
        const float* ray_origin,
        const float* rays,
        size_t num_rays,
        const unsigned int* clBVHindicesOrTriLists,
        const float* clBVHlimits,
        const float* clTriangleIntersectionData,
        const unsigned int* clTriIdxList,
        float* result,
        uint8_t* result_hits
    );


};

} // namespace lvr2

#include <lvr2/algorithm/raycasting/BVHRaycaster.tcc>