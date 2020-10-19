#ifndef __SCANTYPES_HPP__
#define __SCANTYPES_HPP__

#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/registration/CameraModels.hpp"

#include <boost/optional.hpp>
#include <boost/filesystem.hpp>
#include <string_view>

#include <opencv2/core.hpp>

#include <memory>
#include <vector>
#include<string>
namespace lvr2
{

/*****************************************************************************
 * @brief   Struct to hold a Fullwaveform Data for a scan
 * 
 *****************************************************************************/

struct Waveform
{
    Waveform() : maxBucketSize(0),// amplitude(), deviation(), reflectance(), backgroundRadiation(),
    waveformSamples(){};
    ~Waveform(){};
    /// Sensor type flag
    static constexpr char                    sensorType[] = "Waveform";

    /// Max Bucket Size of Waveform samples
    int                                      maxBucketSize;
/*
    /// Amplitude
    std::vector<float>                       amplitude;

    /// Deviation
    std::vector<float>                       deviation;

    /// Reflectance
    std::vector<float>                       reflectance;

    /// Background Radiation
    std::vector<float>                       backgroundRadiation;*/

    //Waveform
    std::vector<uint16_t>                    waveformSamples;
    std::vector<int>                         waveformIndices;
    std::vector<bool>                        lowPower;
    //Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic>         waveformSamples;
};
using WaveformPtr = std::shared_ptr<Waveform>;

/*****************************************************************************
 * @brief Class to represent a scan within a scan project
 ****************************************************************************/
struct Scan
{
    Scan() :
        points(nullptr),
        registration(Transformd::Identity()),
        poseEstimation(Transformd::Identity()),
        thetaMin(0), thetaMax(0),
        phiMin(0), phiMax(0),
        hResolution(0),
        vResolution(0),
        pointsLoaded(false),
        positionNumber(0),
        numPoints(0),
        scanRoot(boost::filesystem::path("./"))
    {}

    ~Scan() {};

    static constexpr char           sensorType[] = "Scan";

    /// Point buffer containing the scan points
    PointBufferPtr                  points;

    /// Containing all Waveform related data
    WaveformPtr                     waveform;

    /// Registration of this scan in project coordinates
    Transformd                      registration;

    /// Pose estimation of this scan in project coordinates
    Transformd                      poseEstimation;

    /// Axis aligned bounding box of this scan
    BoundingBox<BaseVector<float> > boundingBox;

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

    /// Indicates if all points ware loaded from the initial
    /// input file
    bool                            pointsLoaded;

    /// Scan position number of this scan in the current scan project
    int                             positionNumber;

    /// Path to root dir of this scan
    boost::filesystem::path         scanRoot;

    /// Name of the file containing the scan data
    boost::filesystem::path         scanFile;

    /// Number of points in scan
    size_t                          numPoints;
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
    /// Sensor type flag
    static constexpr char           sensorType[] = "ScanImage";

    /// Extrinsics 
    Extrinsicsd                     extrinsics;

    /// Extrinsics estimate
    Extrinsicsd                     extrinsicsEstimate;

    /// Path to stored image
    boost::filesystem::path         imageFile;

    /// OpenCV representation
    cv::Mat                         image;
};



using ScanImagePtr = std::shared_ptr<ScanImage>;
using ScanImageOptional = boost::optional<ScanImage>;


/*****************************************************************************
 * @brief   Represents a camera that was used at a specific scan
 *          position. The intrinsic calibration is stored in the
 *          camera's camera field. Each image has its owen orientation
 *          (extrinsic matrix) with respect to the laser scanner
 * 
 ****************************************************************************/
struct ScanCamera 
{
    /// Description of the sensor model
    static constexpr char           sensorType[] = "ScanCamera";

    /// Individual name of the camera
    std::string                     sensorName = "Camera";

    /// Pinhole camera model
    PinholeModeld                   camera;

    /// Pointer to a set of images taken at a scan position
    std::vector<ScanImagePtr>       images;
};

using ScanCameraPtr = std::shared_ptr<ScanCamera>;


/*****************************************************************************
 * @brief   Struct to hold a camera hyperspectral panorama
 *          together with a timestamp
 * 
 *****************************************************************************/

struct HyperspectralPanoramaChannel
{
    /// Sensor type flag
    static constexpr char           sensorType[] = "HyperspectralPanoramaChannel";

    /// Timestamp 
    double                          timestamp;

    /// Path to stored image
    boost::filesystem::path         channelFile;

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

struct HyperspectralPanorama
{
    /// Sensor type flag
    static constexpr char                          sensorType[] = "HyperspectralPanorama";

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

struct HyperspectralCameraModel
{
    /// Sensor type flag
    static constexpr char           sensorType[] = "HyperspectralCameraModel";

    /// Extrinsics 
    Extrinsicsd                     extrinsics;

    /// Extrinsics estimate
    Extrinsicsd                     extrinsicsEstimate;

    /// Focal length
    double                          focalLength;

    /// Offset angle
    double                          offsetAngle;

    /// Principal x, y, z
    Vector3d                        principal;

    /// Distortion
    Vector3d                        distortion;
};

using HyperspectralCameraModelPtr = std::shared_ptr<HyperspectralCameraModel>;

/*****************************************************************************
 * @brief   Struct to hold a hyperspectral camera
 *          together with it's camera model and panoramas
 * 
 *****************************************************************************/

struct HyperspectralCamera
{
    /// Sensor type flag
    static constexpr char                    sensorType[] = "HyperspectralCamera";

    /// Camera model
    // HyperspectralCameraModelPtr              cameraModel;

    /// Extrinsics 
    Extrinsicsd                              extrinsics;

    /// Extrinsics estimate
    Extrinsicsd                              extrinsicsEstimate;

    /// Focal length
    double                                   focalLength;

    /// Offset angle
    double                                   offsetAngle;

    /// Principal x, y, z
    Vector3d                                 principal;

    /// Distortion
    Vector3d                                 distortion;

    /// OpenCV representation
    std::vector<HyperspectralPanoramaPtr>    panoramas;
};

using HyperspectralCameraPtr = std::shared_ptr<HyperspectralCamera>;
/*****************************************************************************
 * @brief   Struct to represent a LabelInstance
 *****************************************************************************/
struct LabelInstance
{
    static constexpr char           sensorType[] = "LabelInstance";
    std::string instanceName;

    Vector3i color;

    std::vector<int> labeledIDs;
};
using LabelInstancePtr = std::shared_ptr<LabelInstance>;
/*****************************************************************************
 * @brief   Struct to represent a LabelClass
 *****************************************************************************/
struct LabelClass
{
    static constexpr char           sensorType[] = "LabelClass";
    std::string className;

    std::vector<LabelInstancePtr> instances;
};
using LabelClassPtr = std::shared_ptr<LabelClass>;
/*****************************************************************************
 * @brief   Struct to represent a LabelRoot
 *****************************************************************************/
struct LabelRoot
{
    static constexpr char           sensorType[] = "LabelRoot";
    PointBufferPtr points;

    //std::vector<std::pair<uint32_t,uint32_t>,

    std::vector<LabelClassPtr> labelClasses;
};
using LabelRootPtr = std::shared_ptr<LabelRoot>;
/*****************************************************************************
 * @brief   Represents a scan position consisting of a scan and
 *          images taken at this position
 * 
 ****************************************************************************/
struct ScanPosition
{
    static constexpr char           sensorType[] = "ScanPosition";

    /// Vector of scan data. The scan position can contain several 
    /// scans. The scan with the best resolition should be stored in
    /// scans[0]. Scans can be empty
    std::vector<ScanPtr>            scans;

    /// Image data (optional, empty vector of no images were taken) 
    std::vector<ScanCameraPtr>      cams;

    /// Image data (optional, empty vector of no hyperspactral panoramas were taken) 
    HyperspectralCameraPtr          hyperspectralCamera;

    /// Latitude (optional)
    double                          latitude = 0.0;

    /// Longitude (optional)        
    double                          longitude = 0.0;

    /// Longitude (optional)        
    double                          altitude = 0.0;

    /// Estimated pose
    Transformd                      pose_estimate;

    /// Final registered position in project coordinates
    Transformd                      registration;

    /// Timestamp when this position was created
    double                          timestamp = 0.0;

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
    /// Type of used laser scanner
    static constexpr char           sensorType[] = "ScanProject";

    /// Individual name of used laser scanner
    std::string                     sensorName;

    /// Position of this scan project in world coordinates.
    /// It is assumed that all stored scan positions are in 
    /// project coordinates
    Transformd                      pose;

    /// Vector of scan positions for this project
    std::vector<ScanPositionPtr>    positions;

    /// Description (tag) of the internally used coordinate
    /// system. It is assumed that all coordinate systems 
    /// loaded with this software are right-handed
    std::string                     coordinateSystem;


};


using ScanProjectPtr = std::shared_ptr<ScanProject>;

/*****************************************************************************
 * @brief   Struct to represent a scan project with marker showing if a scan
 *          pose has been changed
 *****************************************************************************/
struct ScanProjectEditMark
{
    ScanProjectEditMark(){}
    ScanProjectEditMark(ScanProjectPtr _project):project(project){}
    ScanProjectPtr project;
    /// True if scan pose has been changed, one bool for each scan position
    std::vector<bool> changed;
};
using ScanProjectEditMarkPtr = std::shared_ptr<ScanProjectEditMark>;

/*****************************************************************************
 * @brief   Struct to Represt a scan with Labels
 *****************************************************************************/
struct LabeledScanProjectEditMark
{
    LabeledScanProjectEditMark(){}
    LabeledScanProjectEditMark(ScanProjectPtr _project)
    {
        editMarkProject = ScanProjectEditMarkPtr(new ScanProjectEditMark(_project));
    }
    ScanProjectEditMarkPtr editMarkProject;
    
    //Contains all data assoicated with Label
    LabelRootPtr                    labelRoot;
};
using LabeledScanProjectEditMarkPtr = std::shared_ptr<LabeledScanProjectEditMark>;

} // namespace lvr2

#endif
