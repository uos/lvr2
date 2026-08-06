#include "lvr2/algorithm/raycasting/EmbreeRaycaster.hpp"
#include "lvr2/algorithm/pmp/DistancePointTriangle.h"

#if LVR2_EMBREE_VERSION == 3
#include <embree3/rtcore.h>
#else
#include <embree4/rtcore.h>
#endif

namespace lvr2 {

void EmbreeErrorFunction(void* userPtr, enum RTCError error, const char* str)
{
    printf("error %d: %s\n", error, str);
}

bool embreeClosestPointCallback(RTCPointQueryFunctionArguments* args)
{
    auto* state = static_cast<EmbreeClosestPointState*>(args->userPtr);
    const unsigned int primID = args->primID;
    const unsigned int* f = state->faces + primID * 3;
    const float* v = state->vertices;

    pmp::Point a(v[f[0]*3+0], v[f[0]*3+1], v[f[0]*3+2]);
    pmp::Point b(v[f[1]*3+0], v[f[1]*3+1], v[f[1]*3+2]);
    pmp::Point c(v[f[2]*3+0], v[f[2]*3+1], v[f[2]*3+2]);
    pmp::Point q(args->query->x, args->query->y, args->query->z);

    pmp::Point nearest_point;
    pmp::Scalar d = pmp::dist_point_triangle(q, a, b, c, nearest_point);

    if (d < args->query->radius) {
        args->query->radius = static_cast<float>(d);
        state->result.point = nearest_point.cast<float>();
        state->result.face  = FaceHandle(primID);
        state->found = true;
        return true;
    }
    return false;
}

} // namespace lvr2
