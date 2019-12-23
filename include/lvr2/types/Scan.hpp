#ifndef SCAN_HPP_
#define SCAN_HPP_

#include "MatrixTypes.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/BoundingBox.hpp"

#include <iostream>
#include <boost/filesystem.hpp>

#include <Eigen/Dense>

namespace lvr2
{

/***
 * @brief Class to represent a scan within a scan project
 */
struct Scan
{
    Scan() :
        m_points(nullptr),
        m_registration(Transformd::Identity()),
        m_poseEstimation(Transformd::Identity()),
        m_hFieldOfView(0),
        m_vFieldOfView(0),
        m_hResolution(0),
        m_vResolution(0),
        m_pointsLoaded(false),
        m_positionNumber(0),
        m_scanRoot(boost::filesystem::path("./"))
    {}

    ~Scan() {};

    /// Point buffer containing the scan points
    PointBufferPtr                  m_points;

    /// Registration of this scan in project coordinates
    Transformd                      m_registration;

    /// Pose estimation of this scan in project coordinates
    Transformd                      m_poseEstimation;

    /// Axis aligned bounding box of this scan
    BoundingBox<BaseVector<float> > m_boundingBox;

    /// Horizontal field of view of used laser scanner
    float                           m_hFieldOfView;

    /// Vertical field of view of used laser scanner
    float                           m_vFieldOfView;

    /// Horizontal resolution of used laser scanner
    float                           m_hResolution;

    /// Vertical resolution of used laser scanner
    float                           m_vResolution;

    /// Indicates if all points ware loaded from the initial
    /// input file
    bool                            m_pointsLoaded;

    /// Scan position number of this scan in the current scan project
    int                             m_positionNumber;

    /// Path to root dir of this scan
    boost::filesystem::path         m_scanRoot;
};

using ScanPtr = std::shared_ptr<Scan>;

void parseSLAMDirectory(std::string dir, std::vector<ScanPtr>& scans);

} // namespace lvr2
#endif /* !SCAN_HPP_ */
