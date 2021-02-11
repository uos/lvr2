#ifndef LVR2_TYPES_SCANTYPES_HPP
#define LVR2_TYPES_SCANTYPES_HPP

#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/types/CameraModels.hpp"

#include <boost/optional.hpp>
#include <boost/filesystem.hpp>
#include <string_view>

#include <opencv2/core.hpp>

#include <memory>
#include <vector>
#include <string>

namespace lvr2
{

// Forward Declarations 

// Groups
struct ScanProjectType;
struct ScanPositionType;

struct SensorType;
// Abstract Sensor?
using SensorPtr = std::shared_ptr<SensorType>;
struct SensorDataType;
using SensorDataPtr = std::shared_ptr<SensorDataType>;

// Container types
struct ScanProject;
struct ScanPosition;
// Sensor types
struct LIDAR;
struct Camera;
struct HyperspectralCamera;

// Sensor recording types
struct Scan; // LIDAR has N Scans

struct CameraImage; // Camera has N CameraImages

// shared ptr typedefs
using ScanProjectPtr = std::shared_ptr<ScanProject>;
using ScanPositionPtr = std::shared_ptr<ScanPosition>;

using LIDARPtr  = std::shared_ptr<LIDAR>;
using CameraPtr = std::shared_ptr<Camera>;
using HyperspectralCameraPtr = std::shared_ptr<HyperspectralCamera>;

using ScanPtr = std::shared_ptr<Scan>;
using CameraImagePtr = std::shared_ptr<CameraImage>;

struct ScanProjectType {
    static constexpr char           type[] = "ScanProject";
};

struct ScanPositionType {
    static constexpr char           type[] = "ScanPosition";
};

struct SensorType {
    static constexpr char           type[] = "Sensor";
    // Optional name of this sensor: e.g. RieglVZ-400i
    std::string name;
    // Fixed transformation to upper frame (ScanPosition)
};

struct SensorDataType {
    static constexpr char           type[] = "SensorData";
};

struct Transformable {
    /// Transformation to the upper frame:
    /// ScanProject: to World (GPS)
    /// ScanPosition: to ScanProject
    /// Sensor: to ScanPosition
    /// SensorData: to Sensor
    Transformd transformation;
};

/*****************************************************************************
 * @brief   Struct to represent a scan project consisting
 *          of a set of scan position. Each scan position 
 *          can consist of a laser scan and an set of acquired
 *          images. All scan position are numbered incrementally.
 *          If an optional for a scan position returns false,
 *          the corresponding data is not available for this 
 *          scan position number.
 *****************************************************************************/
struct ScanProject : ScanProjectType, Transformable
{
    //// META BEGIN

    // the project coordinate system. This is specified as a combination of a geo-coordinate system (as EPSG reference)
    std::string                     crs;

    // and a 4x4 transformation matrix that will be applied to all data in the project to convert it from the local (project specific) 
    // coordinate into the geo-coordinate system
    // transformation lies in "Transformable base class"
    // Transformd                      transformation;

    // optional name of ScanProject
    std::string                     name;

    /// Description (tag) of the internally used coordinate
    /// system. It is assumed that all coordinate systems 
    /// loaded with this software are right-handed
    std::string                     coordinateSystem = "right-handed";
    std::string                     unit = "m";

    //// META END

    //// HIERARCHY BEGIN

    /// Vector of scan positions for this project
    std::vector<ScanPositionPtr>    positions;

    //// HIERARCHY END
};

struct ScanPosition : ScanPositionType, Transformable
{
    /// META BEGIN

    /// Estimated pose relativ to upper coordinate system
    Transformd                         poseEstimation;

    /// Final registered position in project coordinates (relative to upper coordinate system: e.g. ScanProject)
    // Transformd                      transformation;

    /// Timestamp when this position was created
    double                             timestamp = 0.0;

    //// META END

    //// HIERARCHY BEGIN
    // Sensors applied to a ScanPosition

    // 1. LIDARs
    std::vector<LIDARPtr>               lidars;

    /// 2. Cameras
    std::vector<CameraPtr>              cameras;

    /// 3. Hyperspectral Cameras
    std::vector<HyperspectralCameraPtr> hyperspectral_cameras;

    //// HIERARCHY END
};

/*****************************************************************************
 * @brief   Represents a LIDAR sensor that was used at a specific scan
 *          position. The intrinsic parameters are stored in here. Extrinsic parameters
 *          are relative to the upper coordinate system: ScanPosition.
 *          Most of the time these 
 * 
 ****************************************************************************/

// Flag Struct
struct LIDAR : SensorType, Transformable
{
    //// META BEGIN
    // TODO: check boost type_info
    static constexpr char           kind[] = "LIDAR";
    //// META END

    //// HIERARCHY BEGIN
    ///  Scans recorded from this LIDAR sensor
    std::vector<ScanPtr> scans;

    //// HIERARCHY END
};

/*****************************************************************************
 * @brief   Represents a camera that was used at a specific scan
 *          position. The intrinsic calibration is stored in the
 *          camera's camera field. Each image has its owen orientation
 *          (extrinsic matrix) with respect to the laser scanner
 * 
 ****************************************************************************/
struct Camera : SensorType, Transformable
{
    //// META BEGIN
    // TODO: check boost::typeindex<>::pretty_name (contains lvr2 as namespace: "lvr2::Camera")
    static constexpr char             kind[] = "Camera";
    /// Pinhole camera model
    PinholeModel                      model;
    //// META END
    //// HIERARCHY BEGIN
    /// Pointer to a set of images taken at a scan position
    std::vector<CameraImagePtr>       images;
    //// HIERARCHY END
};

/*****************************************************************************
 * @brief Struct to represent a scan within a scan project
 ****************************************************************************/
struct Scan : SensorDataType, Transformable
{
    //// META BEGIN
    static constexpr char           kind[] = "Scan";

    /// Dynamic transformation of this sensor data
    /// Example 1:
    /// per scan position we have an old Sick Scanner rotating 
    /// around. Thus, this scanner acquires different scans at
    /// one fixed scan position but with dynamic transformations
    /// of each scan
    /// Variable "transform is placed in Transformable"
    // Transformd                       transform;

    /// Pose estimation of this scan in project coordinates
    Transformd                       poseEstimation;

    /// Min horizontal scan angle
    double                           thetaMin;

    /// Max horizontal scan angle
    double                           thetaMax;

    /// Min vertical scan angle
    double                           phiMin;

    /// Max vertical scan angle
    double                           phiMax;

    /// Horizontal resolution of used laser scanner
    double                           hResolution;

    /// Vertical resolution of used laser scanner
    double                           vResolution;

    /// Start timestamp 
    double                           startTime;

    /// End timestamp     
    double                           endTime;

    /// Number of points in scan
    size_t                           numPoints;

    /// Axis aligned bounding box of this scan
    BoundingBox<BaseVector<float> >  boundingBox;

    //// META END

    /// Point buffer containing the scan points
    PointBufferPtr                   points;
};

/*****************************************************************************
 * @brief   Struct to hold a camera image together with intrinsic 
 *          and extrinsic camera parameters
 * 
 *****************************************************************************/

struct CameraImage : SensorDataType, Transformable
{
    static constexpr char           kind[] = "CameraImage";
    /// Extrinsics estimate
    Extrinsicsd                     extrinsicsEstimation;

    // /// Extrinsics : Is not transformation. See Transformable
    // Extrinsicsd                     extrinsics;

    /// OpenCV representation
    cv::Mat                         image;

    /// Timestamp 
    double                          timestamp;
};



/*****************************************************************************
 * @brief   Struct to hold a camera hyperspectral panorama
 *          together with a timestamp
 * 
 *****************************************************************************/

struct HyperspectralPanoramaChannel : SensorDataType
{
    static constexpr char           kind[]  =  "HyperspectralPanoramaChannel";

    /// Timestamp 
    double                          timestamp;

    /// wavelength
    double                          wavelength;

    /// wavelength inverval?
    // double                          wavelength[2];

    /// OpenCV representation
    cv::Mat                         channel;
};

using HyperspectralPanoramaChannelPtr = std::shared_ptr<HyperspectralPanoramaChannel>;
using HyperspectralPanoramaChannelOptional = boost::optional<HyperspectralPanoramaChannel>;

/*****************************************************************************
 * @brief   Struct to hold a camera hyperspectral panorama
 *          together with a timestamp
 * 
 *****************************************************************************/

struct HyperspectralPanorama : SensorDataType, Transformable
{
    /// Sensor type flag
    static constexpr char                          kind[] = "HyperspectralPanorama";

    /// preview generated from channels (optional: check if preview.empty())
    // cv::Mat                                        preview;

    /// minimum and maximum wavelength
    double                                         wavelength[2];
    /// resolution in x and y
    size_t                                         resolution[2];

    /// OpenCV representation
    std::vector<HyperspectralPanoramaChannelPtr>   channels;
};

using HyperspectralPanoramaPtr = std::shared_ptr<HyperspectralPanorama>;
using HyperspectralPanoramaOptional = boost::optional<HyperspectralPanorama>;

/*****************************************************************************
 * @brief   Struct to hold a hyperspectral camera model
 *          together with intrinsic, extrinsic and further parameters
 * 
 *****************************************************************************/

// /*****************************************************************************
//  * @brief   Struct to hold a hyperspectral camera
//  *          together with it's camera model and panoramas
//  * 
//  *****************************************************************************/

struct HyperspectralCamera : SensorType, Transformable
{
    /// Sensor type flag
    static constexpr char                    kind[] = "HyperspectralCamera";

    /// Camera model
    CylindricalModel                         model;

    /// Extrinsics estimate
    Extrinsicsd                              extrinsicsEstimation;

    /// OpenCV representation
    std::vector<HyperspectralPanoramaPtr>    panoramas;
};


/*****************************************************************************
 * @brief   Represents a scan position consisting of a scan and
 *          images taken at this position
 * 
 ****************************************************************************/





// TODO: HowTo represent Labels?
// Labels at 
// - points
// - images (Rect, pixelwise)
// - faces







// /*****************************************************************************
//  * @brief   Struct to represent a scan project with marker showing if a scan
//  *          pose has been changed
//  *****************************************************************************/
// struct ScanProjectEditMark
// {
//     ScanProjectEditMark(){}
//     ScanProjectEditMark(ScanProjectPtr _project):project(project){}
//     ScanProjectPtr project;
//     /// True if scan pose has been changed, one bool for each scan position
//     std::vector<bool> changed;
// };
// using ScanProjectEditMarkPtr = std::shared_ptr<ScanProjectEditMark>;

// /*****************************************************************************
//  * @brief   Struct to Represt a scan with Labels
//  *****************************************************************************/
// struct LabeledScanProjectEditMark
// {
//     LabeledScanProjectEditMark(){}
//     LabeledScanProjectEditMark(ScanProjectPtr _project)
//     {
//         editMarkProject = ScanProjectEditMarkPtr(new ScanProjectEditMark(_project));
//     }
//     ScanProjectEditMarkPtr editMarkProject;
    
//     //Contains all data assoicated with Label
//     LabelRootPtr                    labelRoot;
// };
// using LabeledScanProjectEditMarkPtr = std::shared_ptr<LabeledScanProjectEditMark>;



} // namespace lvr2

#endif // LVR2_TYPES_SCANTYPES_HPP
