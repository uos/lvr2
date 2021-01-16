#include "lvr2/io/descriptions/HDF5MetaDescriptionV2.hpp"
#include "lvr2/io/yaml/MetaNodeDescriptions.hpp"
#include "lvr2/io/hdf5/Hdf5Util.hpp"

namespace lvr2
{

void HDF5MetaDescriptionV2::saveScanProject(
    HighFive::Group &g,
    const YAML::Node &n) const 
{
    ScanProject sp;
    if(YAML::convert<ScanProject>::decode(n, sp))
    {   
        hdf5util::addAtomic<std::string>(g, "sensorType", ScanProject::sensorType);
        // write data
        hdf5util::addMatrix<double>(g, "poseEstimation", sp.pose);
        hdf5util::addAtomic<std::string>(g, "coordinateSystem", sp.coordinateSystem);
        hdf5util::addAtomic<std::string>(g, "sensorName", sp.sensorName);
    }
}

void HDF5MetaDescriptionV2::saveScanPosition(
    HighFive::Group &g,
    const YAML::Node &n) const
{
    ScanPosition sp;
    if(YAML::convert<ScanPosition>::decode(n, sp))
    {
        hdf5util::addAtomic<std::string>(g, "sensorType", ScanPosition::sensorType);
        // Is ScanPosition
        
        // GPS
        doubleArr gps(new double[3]);
        gps[0] = sp.latitude;
        gps[1] = sp.longitude;
        gps[2] = sp.altitude;
        hdf5util::addArray<double>(g, "gpsPosition", 3, gps);

        // Pose estimation and registration
        hdf5util::addMatrix<double>(g, "poseEstimation", sp.pose_estimate);
        hdf5util::addMatrix<double>(g, "registration", sp.registration);

        // Timestamp
        hdf5util::addAtomic<double>(g, "timestamp", sp.timestamp);
    }
}

void HDF5MetaDescriptionV2::saveScan(
    HighFive::Group &g,
    const YAML::Node &n) const
{
    Scan s;
    if(YAML::convert<Scan>::decode(n, s))
    {
        hdf5util::addAtomic<std::string>(g, "sensorType", Scan::sensorType);

        doubleArr phi(new double[2]);
        phi[0] = s.phiMin;
        phi[1] = s.phiMax;
        hdf5util::addArray(g, "phi", 2, phi);

        doubleArr theta(new double[2]);
        theta[0] = s.thetaMin;
        theta[1] = s.thetaMax;
        hdf5util::addArray(g, "theta", 2, theta);

        doubleArr resolution(new double[2]);
        resolution[0] = s.hResolution;
        resolution[1] = s.vResolution;
        hdf5util::addArray(g, "resolution", 2, resolution);

        doubleArr timestamps(new double[2]);
        timestamps[0] = s.startTime;
        timestamps[1] = s.endTime;
        hdf5util::addArray(g, "timestamps", 2, timestamps);

        hdf5util::addMatrix(g, "poseEstimation", s.poseEstimation);
        hdf5util::addMatrix(g, "registration", s.registration);

        hdf5util::addAtomic(g, "numPoints", s.numPoints);
    }
}

void HDF5MetaDescriptionV2::saveScanCamera(
    HighFive::Group &g,
    const YAML::Node& n) const
{
    ScanCamera sc;
    if(YAML::convert<ScanCamera>::decode(n, sc))
    {
        hdf5util::addAtomic<std::string>(g, "sensorType", ScanCamera::sensorType);

        // Intrinsic Parameters
        doubleArr intrinsics(new double[4]);
        intrinsics[0] = sc.camera.cx;
        intrinsics[1] = sc.camera.cy;
        intrinsics[2] = sc.camera.fx;
        intrinsics[3] = sc.camera.fy;
        hdf5util::addArray(g, "intrinsics", 4, intrinsics);

        // Distortion parameter
        hdf5util::addAtomic<std::string>(g, "distortionModel", sc.camera.distortionModel);
        hdf5util::addVector(g, "distortionParameter", sc.camera.k);

        uintArr res(new unsigned int[2]);
        res[0] = sc.camera.width;
        res[1] = sc.camera.height;
        hdf5util::addArray(g, "resolution", 2, res);
        hdf5util::addAtomic<std::string>(g, "sensorName", sc.sensorName);
    }
}

void HDF5MetaDescriptionV2::saveScanImage(
    HighFive::Group &g,
    const YAML::Node &n) const
{
    ScanImage si;
    if(YAML::convert<ScanImage>::decode(n, si))
    {
        hdf5util::addAtomic<std::string>(g, "sensorType", ScanImage::sensorType);
        hdf5util::addMatrix(g, "extrinsics", si.extrinsics);
        hdf5util::addMatrix(g, "extrinsicsEstimate", si.extrinsicsEstimate);
        hdf5util::addAtomic<std::string>(g, "imageFile", si.imageFile.string());
    }
}

void HDF5MetaDescriptionV2::saveChannel(
    HighFive::DataSet &d,
    const YAML::Node &n) const
{
    std::string attr_name, attr_value;

    attr_name = "sensor_type";
    attr_value = n[attr_name].as<std::string>();
    hdf5util::setAttribute(d, attr_name, attr_value);

    attr_name = "channel_type";
    attr_value = n[attr_name].as<std::string>();
    hdf5util::setAttribute(d, attr_name, attr_value);

    size_t num_elements = n["dims"][0].as<size_t>();
    attr_name = "num_elements";
    hdf5util::setAttribute(d, attr_name, num_elements);
    
    size_t width = n["dims"][1].as<size_t>();
    attr_name = "width";
    hdf5util::setAttribute(d, attr_name, width);
}

void HDF5MetaDescriptionV2::saveHyperspectralCamera(
    HighFive::Group &g,
    const YAML::Node& n) const
{
    // TODO:
}

void HDF5MetaDescriptionV2::saveHyperspectralPanoramaChannel(
    HighFive::Group &g,
    const YAML::Node &n) const
{
    // TODO:
}

YAML::Node HDF5MetaDescriptionV2::scanProject(const HighFive::Group &g) const 
{
    YAML::Node node;

    boost::optional<std::string> sensorTypeOpt = hdf5util::getAtomic<std::string>(g, "sensorType");

    if(sensorTypeOpt && *sensorTypeOpt == ScanProject::sensorType)
    {
        ScanProject sp;

        // sensorName
        boost::optional<std::string> sensorNameOpt = hdf5util::getAtomic<std::string>(g, "sensorName");
        if(sensorNameOpt){ sp.sensorName = *sensorNameOpt; }

        // poseEstimation
        boost::optional<Transformd> poseOpt = hdf5util::getMatrix<Transformd>(g, "poseEstimation");
        if(poseOpt){ sp.pose = *poseOpt; }

        // coordinateSystem
        boost::optional<std::string> coordSysOpt = hdf5util::getAtomic<std::string>(g, "coordinateSystem");
        if(coordSysOpt){ sp.coordinateSystem = *coordSysOpt; }

        node = sp;
    }

    return node;
}

YAML::Node HDF5MetaDescriptionV2::scanPosition(const HighFive::Group &g) const 
{
    YAML::Node node;

    boost::optional<std::string> sensorTypeOpt = hdf5util::getAtomic<std::string>(g, "sensorType");

    if(sensorTypeOpt && *sensorTypeOpt == ScanPosition::sensorType)
    {
        ScanPosition sp;

        // Get GPS information
        size_t dim;
        doubleArr gps = hdf5util::getArray<double>(g, "gpsPosition", dim);
        if(gps && dim == 3)
        {
            sp.latitude = gps[0];
            sp.longitude = gps[1];
            sp.altitude = gps[2];
        }

        // Get timestamp
        boost::optional<double> ts_opt = hdf5util::getAtomic<double>(g, "timestamp");
        if(ts_opt)
        {
            sp.timestamp = *ts_opt;
        }

        // Get pose estimation and registration
        boost::optional<Transformd> poseEstimate = 
            hdf5util::getMatrix<Transformd>(g, "poseEstimation");
        if(poseEstimate)
        {
            sp.pose_estimate = *poseEstimate;
        }
        
        boost::optional<Transformd> registration = 
            hdf5util::getMatrix<Transformd>(g, "registration");
        if(registration)
        {
            sp.registration = *registration;
        }

        node = sp;
    }

    return node;
}

YAML::Node HDF5MetaDescriptionV2::scan(const HighFive::Group &g) const 
{   
    YAML::Node node;
    boost::optional<std::string> sensorTypeOpt = hdf5util::getAtomic<std::string>(g, "sensorType");

    if(sensorTypeOpt && *sensorTypeOpt == Scan::sensorType)
    {
        Scan s;

        // Get start and end time
        size_t dim;
        doubleArr times = hdf5util::getArray<double>(g, "timestamps", dim);
        if(times && dim == 2)
        {
            s.startTime = times[0];
            s.endTime = times[1];
        }

        // Get pose estimation and registration
        boost::optional<Transformd> poseEstimate = 
            hdf5util::getMatrix<Transformd>(g, "poseEstimation");
        if(poseEstimate)
        {
            s.poseEstimation = *poseEstimate;
        }

        boost::optional<Transformd> registration = 
            hdf5util::getMatrix<Transformd>(g, "registration");
        if(poseEstimate)
        {
            s.registration = *registration;
        }

        // Configuration parameters
        
        doubleArr phi = hdf5util::getArray<double>(g, "phi", dim);
        if(phi && dim == 2)
        {
            s.phiMin = phi[0];
            s.phiMax = phi[1];
        } else {
            std::cout << "WARNING: Could not load scan meta 'phi'" << std::endl; 
        }
        
        doubleArr theta = hdf5util::getArray<double>(g, "theta", dim);
        if(theta && dim == 2)
        {
            s.thetaMin = theta[0];
            s.thetaMax = theta[1];
        } else {
            std::cout << "WARNING: Could not load scan meta 'theta'" << std::endl; 
        }

        doubleArr res = hdf5util::getArray<double>(g, "resolution", dim);
        if(res && dim == 2)
        {
            s.hResolution = res[0];
            s.vResolution = res[1];
        }

        // this function doesnt know anything about the "points" location
        auto num_points = hdf5util::getAtomic<size_t>(g, "numPoints");
        if(num_points)
        {
            s.numPoints = *num_points;
        }

        node = s;
    }

    return node;
}

YAML::Node HDF5MetaDescriptionV2::channel(const HighFive::DataSet& d) const
{
    YAML::Node node;
    
    boost::optional<std::string> sensorTypeOpt
        = hdf5util::getAttribute<std::string>(d, "sensor_type");

    if(sensorTypeOpt && *sensorTypeOpt == "Channel")
    {
        node["sensor_type"] = "Channel";
        boost::optional<std::string> channelTypeOpt 
            = hdf5util::getAttribute<std::string>(d, "channel_type");

        if(channelTypeOpt)
        {
            node["channel_type"] = *channelTypeOpt;
        } else {
            // from datasetinfo
            HighFive::DataType dtype = d.getDataType();
            auto lvrTypeName = hdf5util::highFiveTypeToLvr(dtype.string());
            if(lvrTypeName)
            {
                node["channel_type"] = *lvrTypeName;
            } else {
                // Problem: Were certain about dataset to be a channel 
                // but no dimensions? 
                // - I think we should not continue here
                throw std::runtime_error("Could not determine Channel-Type from Hdf5 Dataset");
            }
        }

        boost::optional<size_t> numElementsOpt
            = hdf5util::getAttribute<size_t>(d, "num_elements");

        if(numElementsOpt)
        {
            node["num_elements"] = *numElementsOpt;
        } else {
            // from datasetinfo
            std::vector<size_t> dims = d.getSpace().getDimensions();
            if(dims.size() > 0)
            {
                node["num_elements"] = dims[0];
            } else {
                // Problem: Certain about dataset to be a channel but no dimensions? 
                // - I think we should not continue here
                throw std::runtime_error("Channel in Hdf5 File has no dimensions");
            }
        }

        boost::optional<size_t> widthOpt
            = hdf5util::getAttribute<size_t>(d, "width");

        if(widthOpt)
        {
            node["width"] = *widthOpt;
        } else {
            // from datasetinfo
            // std::cout << "TODO: get Meta width from dataset info" << std::endl;
            std::vector<size_t> dims = d.getSpace().getDimensions();
            if(dims.size() > 1)
            {
                node["width"] = dims[1];
            } else {
                // can do this, because at this point we are 
                // certain about the dataset to be a channel
                node["width"] = 1;
            }
        }

    } else {
        // get meta from dataset itself
        
        HighFive::DataType dtype = d.getDataType();
        HighFive::DataSpace dspace = d.getSpace();
        
        std::vector<size_t> dims = dspace.getDimensions();

        if(dims.size() == 2)
        {
            auto lvrTypeName = hdf5util::highFiveTypeToLvr(dtype.string());
            if(lvrTypeName)
            {
                // could convert datatype
                node["sensor_type"] = "Channel";
                node["channel_type"] = *lvrTypeName;
                node["num_elements"] = dims[0];
                node["width"] = dims[1];
            }
        }
    }

    return node;
}

YAML::Node HDF5MetaDescriptionV2::scanCamera(const HighFive::Group &g) const 
{
    YAML::Node node;

    boost::optional<std::string> sensorTypeOpt 
        = hdf5util::getAtomic<std::string>(g, "sensorType");

    if(!sensorTypeOpt)
    {
        throw std::runtime_error("HDF5MetaDescriptionV2: Could not read sensorType from ScanCamera meta!");
    }

    if(*sensorTypeOpt == ScanCamera::sensorType)
    {
        ScanCamera sc;

        size_t dim;
        doubleArr intrinsics = hdf5util::getArray<double>(g, "intrinsics", dim);
        if(dim == 4)
        {
            sc.camera.cx = intrinsics[0];
            sc.camera.cy = intrinsics[1];
            sc.camera.fx = intrinsics[2];
            sc.camera.fy = intrinsics[3];
        }

        boost::optional<std::string> distModelOpt
            = hdf5util::getAtomic<std::string>(g, "distortionModel");
        if(distModelOpt)
        {
            sc.camera.distortionModel = *distModelOpt;
        }

        auto kOpt = hdf5util::getVector<double>(g, "distortionParameter");
        if(kOpt)
        {
            sc.camera.k = *kOpt;
        }

        uintArr res = hdf5util::getArray<unsigned int>(g, "resolution", dim);
        if(dim == 2)
        {
            sc.camera.width = res[0];
            sc.camera.height = res[1];
        }

        auto sensorNameOpt = hdf5util::getAtomic<std::string>(g, "sensorName");
        if(sensorNameOpt)
        {
            sc.sensorName = *sensorNameOpt;
        }

        node = sc;
    } else {
        std::cout << *sensorTypeOpt << " != "  << ScanCamera::sensorType << std::endl;
        throw std::runtime_error("sensorType differs");
    }

    return node;
}

YAML::Node HDF5MetaDescriptionV2::scanImage(const HighFive::Group &g) const 
{
    YAML::Node node;

    boost::optional<std::string> sensorTypeOpt 
        = hdf5util::getAtomic<std::string>(g, "sensorType");
    
    if(!sensorTypeOpt)
    {
        throw std::runtime_error("HDF5MetaDescriptionV2: Could not read sensorType from meta group!");
    }

    if(*sensorTypeOpt == ScanImage::sensorType)
    {
        ScanImage si;
        
        auto extrinsicsOpt = hdf5util::getMatrix<Extrinsicsd>(g, "extrinsics");
        if(extrinsicsOpt)
        {
            si.extrinsics = *extrinsicsOpt;
        }
        
        auto extrinsicsEstimateOpt = hdf5util::getMatrix<Extrinsicsd>(g, "extrinsicsEstimate");
        if(extrinsicsEstimateOpt)
        {
            si.extrinsicsEstimate = *extrinsicsEstimateOpt;
        }

        auto imageFileOpt = hdf5util::getAtomic<std::string>(g, "imageFile");
        if(imageFileOpt)
        {
            si.imageFile = *imageFileOpt;
        }

        node = si;

    } else {
        std::cout << *sensorTypeOpt << " != "  << ScanImage::sensorType << std::endl;
        throw std::runtime_error("sensorType differs");
    }

    return node;
}

YAML::Node HDF5MetaDescriptionV2::hyperspectralCamera(const HighFive::Group &g) const 
{
    std::cout << timestamp << "HDF5MetaDescriptionV2::hyperspectralCamera() not implemented..." << std::endl;
    YAML::Node node;
    return node;
}

YAML::Node HDF5MetaDescriptionV2::hyperspectralPanoramaChannel(const HighFive::Group &g) const 
{
     std::cout << timestamp << "HDF5MetaDescriptionV2::hyperspectralPanoramaChannel() not implemented..." << std::endl;
    YAML::Node node;
    return node;
}


} // namespace lvr2
