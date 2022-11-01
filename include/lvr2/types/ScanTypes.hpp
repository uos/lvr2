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
#include "lvr2/util/Logging.hpp"

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

    struct CameraImage; // Camera has n image groups
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

    inline std::ostream& operator<<(std::ostream& os, const ScanProject& p)
    {
        os << timestamp << "[ScanProject] ----------------------------------------------------" << std::endl;
        os << timestamp << "[ScanProject] Name: " << p.name << std::endl;
        os << timestamp << "[ScanProject] Coordinate System: " << p.coordinateSystem << std::endl;
        os << timestamp << "[ScanProject] Unit: " << p.unit << std::endl;
        os << timestamp << "[ScanProject] Number of scan positions: " << p.positions.size() << std::endl;
        os << timestamp << "[ScanProject] Transformation: " << std::endl <<  p.transformation << std::endl;
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const ScanProjectPtr p)
    {
        if (p)
        {
            os << *p;
            os << timestamp << "[ScanProject] Pointer Address " << p.get() << std::endl;
        }
        else
        {
            os << timestamp << "[ScanProject] Nullptr" << std::endl;
        }
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const ScanProject& p)
    {
        log << info << "[Scan Project] Name: " << p.name << lvr2::endl;
        log << "[Scan Project] Coordinate System: " << p.coordinateSystem << lvr2::endl;
        log << "[Scan Project] Unit: " << p.unit << lvr2::endl;
        log << "[Scan Project] Number of scan positions: " << p.positions.size() << lvr2::endl;
        log << "[Scan Project] Transformation: " << lvr2::endl <<  p.transformation << lvr2::endl;
        return log;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const ScanProjectPtr p)
    {
        if (p)
        {
            log << *p;
            log << "[Scan Project] Pointer Address " << p.get() << lvr2::endl;
        }
        else
        {
            log << lvr2::warning << "[Scan Project] Nullptr" << lvr2::endl;
        }
        return log;
    }

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

        /// Original name for the riegl scanproject since there could be empty scans
        double original_name = 0;

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

    inline std::ostream& operator<<(std::ostream& os, const ScanPosition& p)
    {
        os << timestamp << "[Scan Position] ---------------------------------------------------" << std::endl;
        os << timestamp << "[Scan Position] Timestamp" << p.timestamp << std::endl;
        os << timestamp << "[Scan Position] Number of cameras: " << p.cameras.size() << std::endl;
        os << timestamp << "[Scan Position] Number of scan positions: " << p.lidars.size() << std::endl;
        os << timestamp << "[Scan Position] Pose estimation: " << p.poseEstimation << std::endl;
        os << timestamp << "[Scan Position] Transformation: " << p.transformation << std::endl;
        return os;
    }
    
    inline std::ostream& operator<<(std::ostream& os, const ScanPositionPtr p)
    {
        if (p)
        {
            os << *p;
            os << timestamp << "[Scan Position] Pointer Address " << p.get() << std::endl;
        }
        else
        {
            os << timestamp << "[Scan Position] Nullptr" << std::endl;
        }
        return os;
    }
 
    inline lvr2::Logger& operator<<(lvr2::Logger& log, const ScanPosition& p)
    {
        log << lvr2::info << "[Scan Position] Timestamp" << p.timestamp << lvr2::endl;
        log << "[Scan Position] Number of cameras: " << p.cameras.size() << lvr2::endl;
        log << "[Scan Position] Number of scan positions: " << p.lidars.size() << lvr2::endl;
        log << "[Scan Position] Pose estimation: " << lvr2::endl << p.poseEstimation << lvr2::endl;
        log << "[Scan Position] Transformation: " << lvr2::endl << p.transformation << lvr2::endl;
        return log;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const ScanPositionPtr p)
    {
        if (p)
        {
            log << *p;
            log << "[Scan Position] Pointer Address " << p.get() << lvr2::endl;
        }
        else
        {
            log << lvr2::warning << "[Scan Position] Nullptr" << lvr2::endl;
        }
        return log;
    }

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

    inline std::ostream& operator<<(std::ostream& os, const LIDAR& l)
    {
        os << timestamp << "[LiDAR] ----------------------------------------------------------" << std::endl;
        os << timestamp << l.model;
        os << timestamp << "[LiDAR] Number of scans: " << l.scans.size() << std::endl;
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const LIDARPtr p)
    {
        if (p)
        {
            os << *p;
            os << timestamp << "[LiDAR] Pointer Address " << p.get() << std::endl;
        }
        else
        {
            os << timestamp << "[LiDAR] Nullptr" << std::endl;
        }
        return os;
    }


    inline lvr2::Logger& operator<<(lvr2::Logger& log, const LIDAR& l)
    {
        log << lvr2::info << l.model;
        log << "[LiDAR] Number of scans: " << l.scans.size() << lvr2::endl;
        return log;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const LIDARPtr p)
    {
        if (p)
        {
            log << *p;
            log << "[LiDAR] Pointer Address " << p.get() << lvr2::endl;
        }
        else
        {
            log << lvr2::error << "[LiDAR] Nullptr" << lvr2::endl;
        }
        return log;
    }

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
        std::vector<CameraImageGroupPtr> groups;

        //// HIERARCHY END
    };

    inline std::ostream& operator<<(std::ostream& os, const Camera& c)
    {
        os << timestamp << "[Camera] ---------------------------------------------------------" << std::endl;
        os << timestamp << "[Camera] Number of image groups: " << c.groups.size() << std::endl;
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const Camera& c)
    {
        log << lvr2::info << "[Camera] Number of image groups: " << c.groups.size() << lvr2::endl;
        return log;
    }

    inline std::ostream& operator<<(std::ostream& os, const CameraPtr p)
    {
        if (p)
        {
            os << *p;
            os << timestamp << "[Camera] Pointer Address " << p.get() << std::endl;
        }
        else
        {
            os << timestamp << "[Camera] Nullptr" << std::endl;
        }
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const CameraPtr p)
    {
        if (p)
        {
            log << *p;
            log << "[Camera] Pointer Address " << p.get() << lvr2::endl;
        }
        else
        {
            log << lvr2::warning << "[Camera] Nullptr" << lvr2::endl;
        }
        return log;
    }

    /*****************************************************************************
 * @brief Struct to represent a scan within a scan project
 ****************************************************************************/
    struct Scan : std::enable_shared_from_this<Scan>, SensorDataEntity, Transformable, BoundedOptional
    {
        static constexpr char type[] = "scan";

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

        std::function<void(ScanPtr)> points_saver;
        std::function<void(ReductionAlgorithmPtr)> points_saver_reduced;

        void save()
        {
            if(loaded())
            {
                points_saver(shared_from_this());
            }
        }

        void save_reduced(ReductionAlgorithmPtr p)
        {
            if(loaded())
            {
                points_saver_reduced(p);
            }
        }

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

    inline std::ostream& operator<<(std::ostream& os, const Scan& s)
    {
        os << timestamp << "[Scan] -----------------------------------------------------------" << std::endl;
        os << timestamp << "[Scan] Number of Points: " << s.numPoints << std::endl;
        os << timestamp << "[Scan] Start time: " << s.startTime << std::endl;
        os << timestamp << "[Scan] End time: " << s.endTime << std::endl;
        os << timestamp << "[Scan] Pose estimation: " << std::endl << s.poseEstimation << std::endl;
        os << timestamp << "[Scan] Transformation: " << std::endl << s.transformation << std::endl;
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const Scan& s)
    {
        log << lvr2::info << "[Scan] Number of Points: " << s.numPoints << lvr2::endl;
        log << "[Scan] Start time: " << s.startTime << lvr2::endl;
        log << "[Scan] End time: " << s.endTime << lvr2::endl;
        log << "[Scan] Pose estimation: " << lvr2::endl << s.poseEstimation;
        log << "[Scan] Transformation: " << lvr2::endl << s.transformation;
        return log;
    }

    inline std::ostream& operator<<(std::ostream& os, const ScanPtr p)
    {
        if (p)
        {
            os << *p;
            os << timestamp << "[Scan] Pointer Address " << p.get() << std::endl;
        }
        else
        {
            os << timestamp << "[Scan] NullPtr" << std::endl;
        }
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const ScanPtr p)
    {
        if (p)
        {
            log << *p;
            log << "[Scan] Pointer Address " << p.get() << lvr2::endl;
        }
        else
        {
            log << warning << "[Scan] NullPtr" << lvr2::endl;
        }
        return log;
    }


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

    inline std::ostream& operator<<(std::ostream& os, const CameraImage& i)
    {
        os << timestamp << "[Camera Image] ----------------------------------------------------" << std::endl;
        os << timestamp << "[Camera Image] Timestamp: " << i.timestamp << std::endl;
        os << timestamp << "[Camera Image] Loaded: " << i.loaded() << std::endl;
        os << timestamp << "[Camera Image] Loadable: " << i.loadable() << std::endl;
        os << timestamp << "[Camera Image] Image dimensions: " << i.image.cols << " x " << i.image.rows << std::endl;
        os << timestamp << "[Camera Image] Extrinsics estimation: " << i.extrinsicsEstimation << std::endl;
        os << timestamp << "[Camera Image] Transformation: " << std::endl << i.transformation << std::endl;
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const CameraImage& i)
    {
        log << lvr2::info << "[Camera Image] Timestamp: " << i.timestamp << lvr2::endl;
        log << "[Camera Image] Loaded: " << i.loaded() << lvr2::endl;
        log << "[Camera Image] Loadable: " << i.loadable() << lvr2::endl;
        log << "[Camera Image] Image dimensions: " << i.image.cols << " x " << i.image.rows << lvr2::endl;
        log << "[Camera Image] Extrinsics estimation: " << i.extrinsicsEstimation << lvr2::endl;
        log << "[Camera Image] Transformation: " << lvr2::endl << i.transformation << lvr2::endl;
        return log;
    }

    inline std::ostream& operator<<(std::ostream& os, const CameraImagePtr p)
    {
        if (p)
        {
            os << *p;
            os << timestamp << "[Camera Image] Pointer Address " << p.get() << std::endl;
        }
        else
        {
            os << timestamp << "[Camera Image] Nullptr" << std::endl;
        }
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger&  log, const CameraImagePtr p)
    {
        if (p)
        {
            log << *p;
            log << "[Camera Image] Pointer Address " << p.get() << lvr2::endl;
        }
        else
        {
            log << lvr2::warning << "[CameraImage] Nullptr" << lvr2::endl;
        }
        return log;
    }


    struct CameraImageGroup : SensorDataGroupEntity, Transformable
    {
        static constexpr char type[] = "camera_images";

        // Data
        std::vector<CameraImagePtr> images;
    };

    inline std::ostream& operator<<(std::ostream& os, const CameraImageGroup& i)
    {
        os << timestamp << "[Camera Image Group] -----------------------------------------------" << std::endl;
        os << timestamp << "[Camera Image Group] Number of images: " << i.images.size() << std::endl;
        os << timestamp << "[Camera Image Group] Transformation: " << std::endl << i.transformation << std::endl;
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const CameraImageGroup& i)
    {
        log << lvr2::info << "[Camera Image Group] Number of images: " << i.images.size() << lvr2::endl;
        log << "[Camera Image Group] Transformation: " << lvr2::endl << i.transformation << lvr2::endl;
        return log;
    }

    inline std::ostream& operator<<(std::ostream& os, const CameraImageGroupPtr p)
    {
        if (p)
        {
            os << *p;
            os << timestamp << "[Camera Image Group] Pointer Address " << p.get() << std::endl;
        }
        else
        {
            os << timestamp << "[Camera Image Group] Nullptr" << std::endl;
        }
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const CameraImageGroupPtr p)
    {
        if (p)
        {
            log << *p;
            log << "[Camera Image Group] Pointer Address " << p.get() << lvr2::endl;
        }
        else
        {
            log << lvr2::warning << "[Camera Image Group] Nullptr" << lvr2::endl;
        }
        return log;
    }

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

    inline std::ostream& operator<<(std::ostream& os, const HyperspectralPanoramaChannel& c)
    {
        os << timestamp << "[Hyperspectral Panorama Channel] -----------------------------------" << std::endl;
        os << timestamp << "[Hyperspectral Panorama Channel] Timestamp: " << c.timestamp << std::endl;
        os << timestamp << "[Hyperspectral Panorama Channel] Image dimensions: " << c.channel.cols << " x " << c.channel.rows << std::endl;
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const HyperspectralPanoramaChannel& c)
    {
        log << lvr2::info << "[Hyperspectral Panorama Channel] Timestamp: " << c.timestamp << lvr2::endl;
        log << "[Hyperspectral Panorama Channel] Image dimensions: " << c.channel.cols << " x " << c.channel.rows << lvr2::endl;
        return log;
    }

    inline std::ostream& operator<<(std::ostream& os, const HyperspectralPanoramaChannelPtr p)
    {
       if (p)
        {
            os << *p;
            os << timestamp << "[Hyperspectral Panorama Channel] Pointer Address " << p.get() << std::endl;
        }
        else
        {
            os << timestamp << "[Hyperspectral Panorama Channel] Nullptr" << std::endl;
        }
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const HyperspectralPanoramaChannelPtr p)
    {
       if (p)
        {
            log << *p;
            log << "[Hyperspectral Panorama Channel] Pointer Address " << p.get() << lvr2::endl;
        }
        else
        {
            log << lvr2::warning << "[Hyperspectral Panorama Channel] Nullptr" << lvr2::endl;
        }
        return log;
    }

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

    inline std::ostream& operator<<(std::ostream& os, const HyperspectralPanorama& p)
    {
        os << timestamp << "[Hyperspectral Panorama] ------------------------------------------" << std::endl;
        os << timestamp << "[Hyperspectral Panorama] Frames resolution: " << p.framesResolution << std::endl;
        os << timestamp << "[Hyperspectral Panorama] Band Axis: " << p.bandAxis << std::endl;
        os << timestamp << "[Hyperspectral Panorama] Frame Axis: " << p.frameAxis << std::endl;
        os << timestamp << "[Hyperspectral Panorama] Data Type: " << p.dataType << std::endl;
        os << timestamp << "[Hyperspectral Panorama] Number of channels: " << p.num_channels << std::endl;
        os << timestamp << "[Hyperspectral Panorama] Channel vector size: " << p.channels.size() << std::endl; 
        os << timestamp << "[Hyperspectral Panorama] Preview Type: " << p.previewType << std::endl;
        os << timestamp << "[Hyperspectral Panorama] Preview dimensions: " << p.preview.cols << " x " << p.preview.rows << std::endl;
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const HyperspectralPanorama& p)
    {
        log << lvr2::info << "[Hyperspectral Panorama] Frames resolution: " << p.framesResolution << lvr2::endl;
        log << "[Hyperspectral Panorama] Band Axis: " << p.bandAxis << lvr2::endl;
        log << "[Hyperspectral Panorama] Frame Axis: " << p.frameAxis << lvr2::endl;
        log << "[Hyperspectral Panorama] Data Type: " << p.dataType << lvr2::endl;
        log << "[Hyperspectral Panorama] Number of channels: " << p.num_channels << lvr2::endl;
        log << "[Hyperspectral Panorama] Channel vector size: " << p.channels.size() << lvr2::endl; 
        log << "[Hyperspectral Panorama] Preview Type: " << p.previewType << lvr2::endl;
        log << "[Hyperspectral Panorama] Preview dimensions: " << p.preview.cols << " x " << p.preview.rows << lvr2::endl;
        return log;
    }

    inline std::ostream& operator<<(std::ostream& os, const HyperspectralPanoramaPtr p)
    {
        if (p)
        {
            os << *p;
            os << timestamp << "[Hyperspectral Panorama] Pointer Address " << p.get() << std::endl;
        }
        else
        {
            os << timestamp << "[Hyperspectral Panorama] Nullptr" << std::endl;
        }
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const HyperspectralPanoramaPtr p)
    {
        if (p)
        {
            log << *p;
            log << "[Hyperspectral Panorama] Pointer Address " << p.get() << lvr2::endl;
        }
        else
        {
            log << lvr2::warning << "[Hyperspectral Panorama] Nullptr" << lvr2::endl;
        }
        return log;
    }

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

    inline std::ostream& operator<<(std::ostream& os, const HyperspectralCamera& c)
    {
        os << timestamp << "[Hyperspectral Camera] --------------------------------------------" << std::endl;
        os << timestamp << "[Hyperspectral Camera] Cylindrical Model: " << c.model << std::endl;
        os << timestamp << "[Hyperspectral Camera] Extrinsics Estimation: " << std::endl << c.extrinsicsEstimation << std::endl;
        os << timestamp << "[Hyperspectral Camera] Number of panoramas: " << c.panoramas.size() << std::endl; 
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const HyperspectralCamera& c)
    {
        log << lvr2::info << "[Hyperspectral Camera] Cylindrical Model: " << c.model << lvr2::endl;
        log << "[Hyperspectral Camera] Extrinsics Estimation: " << lvr2::endl << c.extrinsicsEstimation << lvr2::endl;
        log << "[Hyperspectral Camera] Number of panoramas: " << c.panoramas.size() << lvr2::endl; 
        return log;
    }


    inline std::ostream& operator<<(std::ostream& os, const HyperspectralCameraPtr p)
    {
        if (p)
        {
            os << *p;
            os << timestamp << "[Hyperspectral Camera] Pointer Address " << p.get() << std::endl;
        }
        else
        {
            os << timestamp << "[Hyperspectral Camera] Nullptr" << std::endl;
        }
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const HyperspectralCameraPtr p)
    {
        if (p)
        {
            log << *p;
            log << "[Hyperspectral Camera] Pointer Address " << p.get() << lvr2::endl;
        }
        else
        {
            log << lvr2::warning << "[Hyperspectral Camera] Nullptr" << lvr2::endl;
        }
        return log;
    }


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

    inline std::ostream& operator<<(std::ostream& os, const Waveform& w)
    {
        os << timestamp << "[Waveform] -------------------------------------------------------" << std::endl;
        os << timestamp << "[Waveform] Max Bucket Size" << w.maxBucketSize << std::endl;
        os << timestamp << "[Waveform] Echo Types: " << w.echoType.size() << std::endl;
        os << timestamp << "[Waveform] Waveform Indices: " << w.waveformIndices.size() << std::endl;
        os << timestamp << "[Waveform] Waveform Samples: " << w.waveformSamples.size() << std::endl;
        os << timestamp << "[Waveform] Low Powers: " << w.lowPower.size() << std::endl;
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const Waveform& w)
    {
        log << lvr2::info << "[Waveform] Max Bucket Size" << w.maxBucketSize << lvr2::endl;
        log << "[Waveform] Echo Types: " << w.echoType.size() << lvr2::endl;
        log << "[Waveform] Waveform Indices: " << w.waveformIndices.size() << lvr2::endl;
        log << "[Waveform] Waveform Samples: " << w.waveformSamples.size() << lvr2::endl;
        log << "[Waveform] Low Powers: " << w.lowPower.size() << lvr2::endl;
        return log;
    }


    inline std::ostream& operator<<(std::ostream& os, const WaveformPtr p)
    {
        if (p)
        {
            os << *p;
            os << timestamp << "[Waveform] Pointer Address " << p.get() << std::endl;
        }
        else
        {
            os << timestamp << "[Waveform] Nullptr" << std::endl;
        }
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const WaveformPtr p)
    {
        if (p)
        {
            log << *p;
            log << "[Waveform] Pointer Address " << p.get() << lvr2::endl;
        }
        else
        {
            log << lvr2::warning << "[Waveform] Nullptr" << lvr2::endl;
        }
        return log;
    }

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
