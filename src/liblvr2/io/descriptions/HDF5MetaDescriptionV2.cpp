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
        std::cout << "Saving Coordinate System " << sp.coordinateSystem << std::endl;
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
    std::cout << "scanProject: CREATING YAML NODE" << std::endl;
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

    std::cout << "Finished." << std::endl;

    return node;
}

YAML::Node HDF5MetaDescriptionV2::scanPosition(const HighFive::Group &g) const 
{
    std::cout << "scanPosition: CREATING YAML NODE" << std::endl;
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
    std::cout << "scan: CREATING YAML NODE" << std::endl;
    
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

YAML::Node HDF5MetaDescriptionV2::scanCamera(const HighFive::Group &g) const 
{
    std::cout << timestamp << "HDF5MetaDescriptionV2::scanCamera() not implemented..." << std::endl;
    YAML::Node node;
    return node;
}

YAML::Node HDF5MetaDescriptionV2::scanImage(const HighFive::Group &g) const 
{
    std::cout << timestamp << "HDF5MetaDescriptionV2::scanImage() not implemented..." << std::endl;
    YAML::Node node;
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
