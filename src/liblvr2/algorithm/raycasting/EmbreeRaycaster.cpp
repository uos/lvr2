#include "lvr2/algorithm/raycasting/EmbreeRaycaster.hpp"

#include <embree3/rtcore.h>
#include <algorithm>
#include <iterator>

namespace lvr2 {

void EmbreeErrorFunction(void* userPtr, enum RTCError error, const char* str)
{
    printf("error %d: %s\n", error, str);
}

} // namespace lvr2