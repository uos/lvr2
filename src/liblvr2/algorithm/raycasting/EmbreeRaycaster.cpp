#include "lvr2/algorithm/raycasting/EmbreeRaycaster.hpp"

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

} // namespace lvr2
