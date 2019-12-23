#ifndef __SCANPROJECT_HPP__
#define __SCANPROJECT_HPP__

#include "lvr2/types/Scan.hpp"
#include "lvr2/types/ScanImage.hpp"
#include "lvr2/types/MatrixTypes.hpp"

#include <algorithm>
#include <vector>
#include <boost/optional.hpp>

namespace lvr2
{

using ScanOptional = boost::optional<Scan>;
using ScanImageOptional = boost::optional<ScanImage>;
using ScanPose = std::pair<ScanOptional, ImageOptional>;

/**
 * @brief   Struct to represent a scan project consisting
 *          of a set of scan position. Each scan position 
 *          can consist of a laser scan and an set of acquired
 *          images. All scan position are numbered incrementally.
 *          If an optional for a scan position returns false,
 *          the corresponding data is not available for this 
 *          scan position number.
 */
struct ScanProject
{
    /// Position of this scan project in world coordinates.
    /// It is assumed that all stored scan position are in 
    /// project coordinates
    Transformd          m_position;

    /// Vector of scan poses for this project
    vector<ScanPose>    m_scanPoses;
};

} // namespace lvr2

#endif