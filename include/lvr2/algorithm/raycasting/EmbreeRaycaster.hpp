
#ifndef LVR2_ALGORITHM_RAYCASTING_EMBREERAYCASTER
#define LVR2_ALGORITHM_RAYCASTING_EMBREERAYCASTER

#include <embree3/rtcore.h>
#include <stdio.h>

#include "lvr2/algorithm/raycasting/RaycasterBase.hpp"
#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/geometry/BaseVector.hpp"

namespace lvr2 {

void errorFunction(void* userPtr, enum RTCError error, const char* str)
{
    printf("error %d: %s\n", error, str);
}

template<typename PointT, typename NormalT>
class EmbreeRaycaster : public RaycasterBase<PointT, NormalT > {
public:
    EmbreeRaycaster(const MeshBufferPtr mesh);
    ~EmbreeRaycaster();

    bool castRay(
        const PointT& origin,
        const NormalT& direction,
        PointT& intersection
    );

    void castRays(
        const PointT& origin,
        const std::vector<NormalT >& directions,
        std::vector<PointT >& intersections,
        std::vector<uint8_t>& hits
    );

    void castRays(
        const std::vector<PointT >& origins,
        const std::vector<NormalT >& directions,
        std::vector<PointT >& intersections,
        std::vector<uint8_t>& hits
    );

protected:

    RTCDevice initializeDevice();
    RTCScene initializeScene(RTCDevice device, const MeshBufferPtr mesh);

    inline RTCRayHit lvr2embree(const PointT& origin, const NormalT& direction) const
    {
        RTCRayHit rayhit;
        rayhit.ray.org_x = origin.x;
        rayhit.ray.org_y = origin.y;
        rayhit.ray.org_z = origin.z;
        rayhit.ray.dir_x = direction.x;
        rayhit.ray.dir_y = direction.y;
        rayhit.ray.dir_z = direction.z;
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