#ifndef LVR2_EXAMPLES_SCANTYPES_DUMMIES_HPP
#define LVR2_EXAMPLES_SCANTYPES_DUMMIES_HPP

#include "lvr2/types/ScanTypes.hpp"

namespace lvr2 {

/**
 * @brief Apply a transformation to a pointbuffer.
 * 
 * @param T 
 * @param points 
 * @return
 */
PointBufferPtr operator*(Transformd T, PointBufferPtr points);

/**
 * @brief Generates a dummy scan projects with some sensors and sensordata
 * - pointclouds of size 2502 in shapes of spheres 
 * - lvr2 logo as images
 * 
 * @return ScanProjectPtr 
 */
ScanProjectPtr dummyScanProject();

} // namespace lvr2

#endif // LVR2_EXAMPLES_SCANTYPES_DUMMIES_HPP