
#ifndef LVR2_ALGORITHM_RAYCASTING_EMBREERAYCASTER
#define LVR2_ALGORITHM_RAYCASTING_EMBREERAYCASTER

#include <embree3/rtcore.h>
#include <stdio.h>

#include "lvr2/algorithm/raycasting/RaycasterBase.hpp"
#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "Intersection.hpp"

namespace lvr2 {

void EmbreeErrorFunction(void* userPtr, enum RTCError error, const char* str);

template<typename IntT>
class EmbreeRaycaster : public RaycasterBase<IntT> {
public:
    EmbreeRaycaster(const MeshBufferPtr mesh);
    ~EmbreeRaycaster();

    bool castRay(
        const Vector3f& origin,
        const Vector3f& direction,
        IntT& intersection);

    // template<typename T>
    // bool castRay(
    //     const Vector3f& origin,
    //     const Vector3f& direction,
    //     T& intersection
    // ){
    //     RTCRayHit rayhit = lvr2embree(origin, direction);
    //     rtcIntersect1(m_scene, &m_context, &rayhit);
        
    //     if constexpr(intersection.template has<intelem::Point>())
    //     {
    //         intersection.point.x() = rayhit.ray.org_x + rayhit.ray.tfar * rayhit.ray.dir_x;
    //         intersection.point.y() = rayhit.ray.org_y + rayhit.ray.tfar * rayhit.ray.dir_y;
    //         intersection.point.z() = rayhit.ray.org_z + rayhit.ray.tfar * rayhit.ray.dir_z;
    //     }

    //     if constexpr(intersection.template has<intelem::Distance>())
    //     {
    //         intersection.dist = rayhit.ray.tfar;
    //     }

    //     if constexpr(intersection.template has<intelem::Normal>())
    //     {
    //         intersection.normal.x() = rayhit.hit.Ng_x;
    //         intersection.normal.y() = rayhit.hit.Ng_y;
    //         intersection.normal.z() = rayhit.hit.Ng_z;
    //     }

    //     if constexpr(intersection.template has<intelem::Face>())
    //     {
    //         intersection.face_id = rayhit.hit.primID;
    //     }

    //     if constexpr(intersection.template has<intelem::Barycentrics>())
    //     {
    //         intersection.b_uv.x() = rayhit.hit.u;
    //         intersection.b_uv.y() = rayhit.hit.v;
    //     }

    //     if constexpr(intersection.template has<intelem::Mesh>())
    //     {
    //         intersection.mesh_id = rayhit.hit.geomID;
    //     }

    //     return (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);
    // }

    // bool castRay(
    //     const Vector3f& origin,
    //     const Vector3f& direction,
    //     unsigned char* intersection,
    //     const unsigned int& flags
    // );

    // bool castRay(
    //     const Vector3f& origin,
    //     const Vector3f& direction,
    //     Vector3f& intersection
    // );

    // void castRays(
    //     const Vector3f& origin,
    //     const std::vector<Vector3f>& directions,
    //     std::vector<Vector3f>& intersections,
    //     std::vector<uint8_t>& hits
    // );

    // void castRays(
    //     const std::vector<Vector3f>& origins,
    //     const std::vector<Vector3f>& directions,
    //     std::vector<Vector3f>& intersections,
    //     std::vector<uint8_t>& hits
    // );

protected:

    RTCDevice initializeDevice();
    RTCScene initializeScene(RTCDevice device, const MeshBufferPtr mesh);

    inline RTCRayHit lvr2embree(const Vector3f& origin, const Vector3f& direction) const
    {
        RTCRayHit rayhit;
        rayhit.ray.org_x = origin.x();
        rayhit.ray.org_y = origin.y();
        rayhit.ray.org_z = origin.z();
        rayhit.ray.dir_x = direction.x();
        rayhit.ray.dir_y = direction.y();
        rayhit.ray.dir_z = direction.z();
        rayhit.ray.tnear = 0;
        rayhit.ray.tfar = INFINITY;
        rayhit.ray.mask = 0;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        return rayhit;
    }

    RTCDevice m_device;
    RTCScene m_scene;
    RTCIntersectContext m_context;

};

} // namespace lvr2

#include "EmbreeRaycaster.tcc"

#endif // LVR2_ALGORITHM_RAYCASTING_EMBREERAYCASTER