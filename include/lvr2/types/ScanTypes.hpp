#ifndef __SCANTYPES_HPP__
#define __SCANTYPES_HPP__

#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/registration/PinholeCameraModel.hpp"

#include <boost/optional.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace lvr2
{

/*****************************************************************************
 * @brief Class to represent a scan within a scan project
 ****************************************************************************/
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

/// Shared pointer to scans
using ScanPtr = std::shared_ptr<Scan>;
using ScanOptional = boost::optional<Scan>;

/*****************************************************************************
 * @brief   Struct to hold a camera image together with intrinsic 
 *          and extrinsic camera parameters
 * 
 *****************************************************************************/
struct ScanImage
{
    /// Camera model 
    PinholeCameraModel              camera;

    /// Mount calibration estimate / extrinsic estimate
    Extrinsicsd                     extrinsicsEstmate;

    /// Path to stored image
    boost::filesystem::path         image_file;

    /// OpenCV representation
    cv::Mat                         image;

};

using ScanImagePtr = std::shared_ptr<ScanImage>;
using ScanImageOptional = boost::optional<ScanImage>;


/*****************************************************************************
 * @brief   Represents a scan position consisting of a scan and
 *          images taken at this position
 * 
*****************************************************************************/
struct ScanPosition
{
    /// Scan data (optional)
    ScanOptional                    scan;

    /// Image data (optional, empty vector of no images were taken) 
    std::vector<ScanImagePtr>       images;
};

using ScanPositionPtr = std::shared_ptr<ScanPosition>;

/*****************************************************************************
 * @brief   Struct to represent a scan project consisting
 *          of a set of scan position. Each scan position 
 *          can consist of a laser scan and an set of acquired
 *          images. All scan position are numbered incrementally.
 *          If an optional for a scan position returns false,
 *          the corresponding data is not available for this 
 *          scan position number.
 *****************************************************************************/
struct ScanProject
{
    /// Position of this scan project in world coordinates.
    /// It is assumed that all stored scan position are in 
    /// project coordinates
    Transformd                      pose;

    /// Vector of scan positions for this project
    std::vector<ScanPositionPtr>    positions;
};

using ScanProjectPtr = std::shared_ptr<ScanProject>;

} // namespace lvr2

#endif