#ifndef SCANTYPES
#define SCANTYPES

#include "lvr2/types/PointBuffer.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/types/CameraModels.hpp"
#include "lvr2/types/Variant.hpp"
#include "lvr2/registration/ReductionAlgorithm.hpp"

#include "lvr2/io/kernels/HDF5Kernel.hpp"
#include "lvr2/io/schema/ScanProjectSchema.hpp"

#include <boost/optional.hpp>
#include <boost/filesystem.hpp>
#include <string_view>

#include <opencv2/core.hpp>

#include <memory>
#include <vector>
#include <string>

#include <boost/variant.hpp>
#include <yaml-cpp/yaml.h>

namespace lvr2
{

    // Forward Declarations

    // Groups
    struct ScanProjectEntity;
    struct ScanPositionEntity;

    struct SensorEntity;
    // Abstract Sensor?
    using SensorPtr = std::shared_ptr<SensorEntity>;
    struct SensorDataEntity;
    using SensorDataPtr = std::shared_ptr<SensorDataEntity>;

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
    struct CameraImageGroup;

    // shared ptr typedefs
    using ScanProjectPtr = std::shared_ptr<ScanProject>;
    using ScanPositionPtr = std::shared_ptr<ScanPosition>;

    using LIDARPtr = std::shared_ptr<LIDAR>;
    using CameraPtr = std::shared_ptr<Camera>;
    using HyperspectralCameraPtr = std::shared_ptr<HyperspectralCamera>;

    using ScanPtr = std::shared_ptr<Scan>;
    using CameraImagePtr = std::shared_ptr<CameraImage>;
    using CameraImageGroupPtr = std::shared_ptr<CameraImageGroup>;
    // either one image or one image group
    using CameraImageOrGroup = Variant<CameraImagePtr, CameraImageGroupPtr>;

    struct BaseEntity {
        YAML::Node metadata;
    };

    struct ScanProjectEntity: public BaseEntity
    {
        static constexpr char entity[] = "scan_project";
    };

    struct ScanPositionEntity: public BaseEntity
    {
        static constexpr char entity[] = "scan_position";
    };

    struct SensorEntity: public BaseEntity
    {
        static constexpr char entity[] = "sensor";
        // Optional name of this sensor: e.g. RieglVZ-400i
        std::string name;
        // Fixed transformation to upper frame (ScanPosition)
    };

    struct SensorDataEntity: public BaseEntity
    {
        static constexpr char entity[] = "sensor_data";
    };

    struct SensorDataGroupEntity: public BaseEntity
    {
        static constexpr char entity[] = "sensor_data_group";
    };

    struct LabelDataEntity: public BaseEntity
    {
        static constexpr char entity[] = "label_data";
    };

    struct Transformable
    {
        /// Transformation to the upper frame:
        /// ScanProject: to World (GPS)
        /// ScanPosition: to ScanProject
        /// Sensor: to ScanPosition
        /// SensorData: to Sensor
        Transformd transformation = Transformd::Identity();
    };

    // using AABB = BoundingBox<BaseVector<float> >;

    struct Bounded
    {
        BoundingBox<BaseVector<float>> boundingBox;
    };

    struct BoundedOptional
    {
        boost::optional<BoundingBox<BaseVector<float>>> boundingBox;
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
    struct ScanProject : ScanProjectEntity, Transformable, BoundedOptional
    {
        //// META BEGIN
        // the type of this struct
        static constexpr char type[] = "scan_project";

        // the project coordinate system. This is specified as a combination of a geo-coordinate system (as EPSG reference)
        std::string crs;

        // and a 4x4 transformation matrix that will be applied to all data in the project to convert it from the local (project specific)
        // coordinate into the geo-coordinate system
        // transformation lies in "Transformable base class"
        // Transformd                      transformation;

        // optional name of ScanProject
        std::string name;

        /// Description (tag) of the internally used coordinate
        /// system. It is assumed that all coordinate systems
        /// loaded with this software are right-handed
        std::string coordinateSystem = "right-handed";
        std::string unit = "meter";

        //// META END

        //// HIERARCHY BEGIN

        /// Vector of scan positions for this project
        std::vector<ScanPositionPtr> positions;

        //// HIERARCHY END
    };

    struct ScanPosition : ScanPositionEntity, Transformable, BoundedOptional
    {
        /// META BEGIN
        // the type of this struct
        static constexpr char type[] = "scan_position";
        /// Estimated pose relativ to upper coordinate system
        Transformd poseEstimation = Transformd::Identity();

        /// Final registered position in project coordinates (relative to upper coordinate system: e.g. ScanProject)
        // Transformd                      transformation;

        /// Timestamp when this position was created
        double timestamp = 0.0;

        double original_name = 6;

        //// META END

        //// HIERARCHY BEGIN
        // Sensors applied to a ScanPosition

        // 1. LIDARs
        std::vector<LIDARPtr> lidars;

        /// 2. Cameras
        std::vector<CameraPtr> cameras;

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
    struct LIDAR : SensorEntity, Transformable, BoundedOptional
    {
        //// META BEGIN
        // TODO: check boost type_info
        static constexpr char type[] = "lidar";
        SphericalModel model;
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
    struct Camera : SensorEntity, Transformable
    {
        //// META BEGIN
        // TODO: check boost::typeindex<>::pretty_name (contains lvr2 as namespace: "lvr2::Camera")
        static constexpr char type[] = "camera";
        /// Pinhole camera model
        PinholeModel model;
        //// META END
        //// HIERARCHY BEGIN
        /// Pointer to a set of images taken at a scan position
        // std::vector<CameraImagePtr>       images;
        std::vector<CameraImageOrGroup> images;

        //// HIERARCHY END
    };

    /*****************************************************************************
 * @brief Struct to represent a scan within a scan project
 ****************************************************************************/
    struct Scan : SensorDataEntity, Transformable, BoundedOptional
    {
        //// META BEGIN
        static constexpr char type[] = "scan";

        /// Dynamic transformation of this sensor data
        /// Example 1:
        /// per scan position we have an old Sick Scanner rotating
        /// around. Thus, this scanner acquires different scans at
        /// one fixed scan position but with dynamic transformations
        /// of each scan
        /// Variable "transform is placed in Transformable"
        // Transformd                       transform;

        /// Pose estimation of this scan in project coordinates
        Transformd poseEstimation = Transformd::Identity();

        // model of the scan
        SphericalModelOptional model;

        // double                           vResolution;

        /// Start timestamp
        double startTime;

        /// End timestamp
        double endTime;

        /// Number of points in scan
        size_t numPoints;

        /// Axis aligned bounding box of this scan
        // BoundingBox<BaseVector<float> >  boundingBox;

        //// META END

        /// Point buffer containing the scan points
        PointBufferPtr points;

        /// Loader
        std::function<PointBufferPtr()> points_loader;
        std::function<PointBufferPtr(ReductionAlgorithmPtr)> points_loader_reduced;

        bool loadable() const
        {
            return points_loader ? true : false;
        }

        bool loaded() const
        {
            return points ? true : false;
        }

        void load()
        {
            if (!loaded())
            {
                points = points_loader();
            }
        }

        void load(ReductionAlgorithmPtr red)
        {
            if (!loaded())
            {
                points = points_loader_reduced(red);
            }
        }

        void release()
        {
            points.reset();
        }
    };

    /*****************************************************************************
 * @brief   Struct to hold a camera image together with intrinsic 
 *          and extrinsic camera parameters
 * 
 *****************************************************************************/

    struct CameraImage : SensorDataEntity, Transformable
    {
        static constexpr char type[] = "camera_image";
        /// Extrinsics estimate
        Extrinsicsd extrinsicsEstimation = Extrinsicsd::Identity();

        // /// Extrinsics : Is not transformation. See Transformable
        // Extrinsicsd                     extrinsics;

        /// OpenCV representation
        cv::Mat image;

        /// Loader
        std::function<cv::Mat()> image_loader;

        bool loadable() const
        {
            return image_loader ? true : false;
        }

        bool loaded() const
        {
            return !image.empty();
        }

        void load()
        {
            if (!loaded())
            {
                image = image_loader();
            }
        }

        void release()
        {
            if (loaded())
            {
                image.release();
            }
        }

        /// Timestamp
        double timestamp;
    };

    struct CameraImageGroup : SensorDataGroupEntity, Transformable
    {
        static constexpr char type[] = "camera_images";

        // Data
        std::vector<CameraImageOrGroup> images;
    };

    /*****************************************************************************
 * @brief   Struct to hold a camera hyperspectral panorama
 *          together with a timestamp
 * 
 *****************************************************************************/

    // Not a lvr channel
    struct HyperspectralPanoramaChannel : SensorDataEntity
    {
        static constexpr char type[] = "spectral_panorama_channel";

        /// Timestamp
        double timestamp;

        /// wavelength
        // double                          wavelength;

        /// wavelength inverval?
        // double                          wavelength[2];

        /// OpenCV representation
        cv::Mat channel;
    };

    using HyperspectralPanoramaChannelPtr = std::shared_ptr<HyperspectralPanoramaChannel>;
    using HyperspectralPanoramaChannelOptional = boost::optional<HyperspectralPanoramaChannel>;

    /*****************************************************************************
 * @brief   Struct to hold a camera hyperspectral panorama
 *          together with a timestamp
 * 
 *****************************************************************************/

    struct HyperspectralPanorama : SensorDataEntity, Transformable
    {
        /// Sensor type flag
        static constexpr char type[] = "spectral_panorama";
        static constexpr char kind[] = "SpectralImage";

        /// Camera model
        //CylindricalModel model;

        /// preview generated from channels (optional: check if preview.empty())
        // cv::Mat                                        preview;

        /// minimum and maximum wavelength
        // double                                         wavelength[2];
        /// resolution in x and y

        unsigned int framesResolution[3];
        unsigned int bandAxis;
        unsigned int frameAxis;
        std::string dataType;

        unsigned int panoramaResolution[3];

        /// Camera model (optional): overrides the HyperspectralCamera model
        CylindricalModelOptional model;

        // DATA

        /// preview generated from channels (optional: check if preview.empty())
        cv::Mat preview;
        std::string previewType;

        /// OpenCV representation
        std::vector<HyperspectralPanoramaChannelPtr> channels;

        /// Number of Channels
        size_t num_channels;
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

    struct HyperspectralCamera : SensorEntity, Transformable
    {
        /// Sensor type flag
        static constexpr char type[] = "spectral_camera";

        /// Camera model
        CylindricalModel model;
        /// Extrinsics estimate
        Extrinsicsd extrinsicsEstimation = Extrinsicsd::Identity();

        /// OpenCV representation
        std::vector<HyperspectralPanoramaPtr> panoramas;
    };

    /*****************************************************************************
 * @brief   Struct to hold a Fullwaveform Data for a scan
 * 
 *****************************************************************************/

    struct Waveform : SensorDataEntity
    {
        Waveform() : maxBucketSize(0), // amplitude(), deviation(), reflectance(), backgroundRadiation(),
                     waveformSamples(){};
        ~Waveform(){};
        /// Sensor type flag
        static constexpr char type[] = "waveform";

        /// Max Bucket Size of Waveform samples
        int maxBucketSize;
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
        std::vector<uint16_t> waveformSamples;
        std::vector<long> waveformIndices;
        std::vector<uint8_t> echoType;
        std::vector<bool> lowPower;
        //Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic>         waveformSamples;
    };
    using WaveformPtr = std::shared_ptr<Waveform>;

    /*****************************************************************************
 * @brief   Struct to represent a LabelInstance
 *****************************************************************************/
    struct LabelInstance : LabelDataEntity
    {
        static constexpr char type[] = "label_instance";
        std::string instanceName;

        Vector3i color;

        std::vector<int> labeledIDs;
    };
    using LabelInstancePtr = std::shared_ptr<LabelInstance>;
    /*****************************************************************************
 * @brief   Struct to represent a LabelClass
 *****************************************************************************/
    struct LabelClass : LabelDataEntity
    {
        static constexpr char type[] = "label_class";
        std::string className;

        std::vector<LabelInstancePtr> instances;
    };
    using LabelClassPtr = std::shared_ptr<LabelClass>;
    /*****************************************************************************
 * @brief   Struct to represent a LabelRoot
 *****************************************************************************/
    struct LabelRoot : LabelDataEntity
    {
        static constexpr char type[] = "label_root";
        PointBufferPtr points;
        WaveformPtr waveform;

        std::vector<std::pair<std::pair<uint32_t, uint32_t>, uint32_t>> pointOffsets;

        std::vector<LabelClassPtr> labelClasses;
    };
    using LabelRootPtr = std::shared_ptr<LabelRoot>;

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

    /*****************************************************************************
 * @brief   Struct to represent a scan project with marker showing if a scan
 *          pose has been changed
 *****************************************************************************/
    struct ScanProjectEditMark
    {
        ScanProjectEditMark() = default;
        ScanProjectEditMark(ScanProjectPtr _project) : project(_project) {}
        ScanProjectPtr project;
        FileKernelPtr kernel;
        ScanProjectSchemaPtr schema;

        /// True if scan pose has been changed, one bool for each scan position
        std::vector<bool> changed;
    };
    using ScanProjectEditMarkPtr = std::shared_ptr<ScanProjectEditMark>;

    /*****************************************************************************
 * @brief   Struct to Represt a scan with Labels
 *****************************************************************************/
    struct LabeledScanProjectEditMark
    {
        LabeledScanProjectEditMark() {}
        LabeledScanProjectEditMark(ScanProjectPtr _project)
        {
            editMarkProject = ScanProjectEditMarkPtr(new ScanProjectEditMark(_project));
        }
        ScanProjectEditMarkPtr editMarkProject;

        //Contains all data assoicated with Label
        LabelRootPtr labelRoot;
    };
    using LabeledScanProjectEditMarkPtr = std::shared_ptr<LabeledScanProjectEditMark>;

} // namespace lvr2

#endif // SCANTYPES
