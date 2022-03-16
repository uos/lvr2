
#ifndef LVR2_UTIL_SYNTHETIC_HPP
#define LVR2_UTIL_SYNTHETIC_HPP

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/types/MeshBuffer.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2 {

namespace synthetic {

MeshBufferPtr genSphere(
    int num_long=50,
    int num_lat=50);

PointBufferPtr genSpherePoints(int num_long=50,
    int num_lat=50);

CameraImagePtr genLVRImage();

} // namespace synthetic

} // namespace lvr2

#endif // LVR2_UTIL_SYNTHETIC_HPP