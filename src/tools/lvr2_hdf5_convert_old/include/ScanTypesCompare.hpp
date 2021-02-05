#ifndef LVR2_SCAN_TYPES_COMPARE_HPP

#include "lvr2/types/ScanTypes.hpp"

namespace lvr2 {

bool equal(const float& a, const float& b);
bool equal(const double& a, const double& b);

bool equal(CameraImagePtr si1, CameraImagePtr si2);

bool equal(CameraPtr sc1, CameraPtr sc2);

bool equal(PointBufferPtr p1, PointBufferPtr p2);

bool equal(ScanPtr s1, ScanPtr s2);

bool equal(LIDARPtr l1, LIDARPtr l2);

bool equal(ScanPositionPtr sp1, ScanPositionPtr sp2);

bool equal(ScanProjectPtr sp1, ScanProjectPtr sp2);

} // namespace lvr2

#endif // LVR2_SCAN_TYPES_COMPARE_HPP