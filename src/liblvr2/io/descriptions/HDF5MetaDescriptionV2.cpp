#include "lvr2/io/descriptions/HDF5MetaDescriptionV2.hpp"
#include "lvr2/io/yaml/MetaNodeDescriptions.hpp"
#include "lvr2/io/hdf5/Hdf5Util.hpp"

namespace lvr2
{

void HDF5MetaDescriptionV2::saveHyperspectralCamera(
    HighFive::Group &g,
    const YAML::Node& n) const
{

}

void HDF5MetaDescriptionV2::saveHyperspectralPanoramaChannel(
    HighFive::Group &g,
    const YAML::Node &n) const
{

}

void HDF5MetaDescriptionV2::saveScan(
    HighFive::Group &g,
    const YAML::Node &n) const
{
    YAML::Node config;
    config = n["config"];
    
    vector<size_t> dim = {2, 1};

    // Phi min/max
    doubleArr phi(new double[2]);
    phi[0] = 0.0;
    phi[1] = 0.0;
    if(config["phi"] && config["phi"].size() == 2)
    {
        phi[0] = config["phi"][0].as<double>();
        phi[1] = config["phi"][1].as<double>();
    }
    hdf5util::addArray<double>(g, "phi", dim, phi);

    // Theta min/max
    doubleArr theta(new double[2]);
    theta[0] = 0.0;
    theta[1] = 0.0;
    if(config["theta"] && config["theta"].size() == 2)
    {
        theta[0] = config["theta"][0].as<double>();
        theta[1] = config["theta"][1].as<double>();
    }
    hdf5util::addArray<double>(g, "theta", dim, theta);

    // Horizontal and vertical resolution
    doubleArr resolution(new double[2]);
    resolution[0] = 0.0;
    resolution[1] = 0.0;
    if(config["h_res"])
    {
        resolution[0] = config["h_res"].as<double>();
    }
    if(config["v_res"])
    {
        resolution[1] = config["v_res"].as<double>();
    }
    hdf5util::addArray<double>(g, "resolution", dim, resolution);

    // Pose estimation and registration
    Transformd p_transform;
    if(n["pose_estimate"])
    {
        p_transform = n["pose_estimate"].as<Transformd>();
    }
    hdf5util::addMatrix<double>(g, "poseEstimation", p_transform);

    Transformd r_transform;
    if(n["registration"])
    {
        r_transform = n["registration"].as<Transformd>();
    }
    hdf5util::addMatrix<double>(g, "registration", r_transform);

    // Timestamps
    doubleArr timestamps(new double[2]);
    timestamps[0] = 0.0;
    timestamps[1] = 0.0;
    if(n["start_time"])
    {
        timestamps[0] = n["start_time"].as<double>();
    }
    if(n["end_time"])
    {
        timestamps[1] = n["end_time"].as<double>();
    }
    hdf5util::addArray<double>(g, "timestamps", dim, timestamps);

}

void HDF5MetaDescriptionV2::saveScanPosition(
    HighFive::Group &g,
    const YAML::Node &n) const
{
     // GPS position
    doubleArr gps(new double[3]);
    gps[0] = 0.0;
    gps[1] = 0.0;
    gps[2] = 0.0;
    if(n["latitude"])
    {
        gps[0] = n["latitude"].as<double>();
    }
    if(n["longitude"])
    {
        gps[1] = n["longitude"].as<double>();
    }
    if(n["altitude"])
    {
        gps[1] = n["altitude"].as<double>();
    }
    hdf5util::addArray<double>(g, "gpsPosition", 3, gps);

    // Pose estimation and registration
    Transformd p_transform;
    if(n["pose_estimate"])
    {
        p_transform = n["pose_estimate"].as<Transformd>();
    }
    hdf5util::addMatrix<double>(g, "poseEstimation", p_transform);

    Transformd r_transform;
    if(n["registration"])
    {
        r_transform = n["registration"].as<Transformd>();
    }
    hdf5util::addMatrix<double>(g, "registration", r_transform);

    // Timestamp
    doubleArr ts(new double[1]);
    ts[0] = 0.0;
    if(n["timestamp"])
    {
        ts[0] = n["timestamp"].as<double>();
    }
    hdf5util::addArray<double>(g, "timestamp", 1, ts);
}

void HDF5MetaDescriptionV2::saveScanProject(
    HighFive::Group &g,
    const YAML::Node &n) const 
{
    std::cout << timestamp << "HDF5MetaDescriptionV2::saveScanProject() not implemented..." << std::endl;

}

void HDF5MetaDescriptionV2::saveScanCamera(
    HighFive::Group &g,
    const YAML::Node& n) const
{
    std::cout << timestamp << "HDF5MetaDescriptionV2::saveScanCamera() not implemented..." << std::endl;
}

void HDF5MetaDescriptionV2::saveScanImage(
    HighFive::Group &g,
    const YAML::Node &n) const
{
    std::cout << timestamp << "HDF5MetaDescriptionV2::saveScanImage() not implemented..." << std::endl;
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

YAML::Node HDF5MetaDescriptionV2::scan(const HighFive::Group &g) const 
{
    YAML::Node node;

    // Get start and end time
    std::vector<size_t> timesDim;
    doubleArr times = hdf5util::getArray<double>(g, "timestamps", timesDim);
    if(times && timesDim.size() == 2 && timesDim[0] == 2 && timesDim[1] == 1)
    {
        std::cout << timestamp << "YAML timestamp..." << std::endl;
        node["start_time"] = times[0]; 
        node["end_time"] = times[1];
    }

    // Get pose estimation and registration
    boost::optional<Transformd> poseEstimate = 
        hdf5util::getMatrix<Transformd>(g, "poseEstimation");
    if(poseEstimate)
    {
        node["pose_estimate"] = *poseEstimate;
    }

    boost::optional<Transformd> registration = 
        hdf5util::getMatrix<Transformd>(g, "registration");
    if(poseEstimate)
    {
        node["registration"] = *registration;
    }

    // Configuration parameters
    YAML::Node config;
    std::vector<size_t> resDim;
    doubleArr phi = hdf5util::getArray<double>(g, "phi", resDim);
    if(phi && resDim.size() == 2 && resDim[0] == 2 && resDim[1] == 1)
    {
        std::cout << timestamp << "YAML phi..." << std::endl;
        config["phi"] = YAML::Load("[]");
        config["phi"].push_back(phi[0]);
        config["phi"].push_back(phi[1]);
    }
    resDim.clear();

    doubleArr theta = hdf5util::getArray<double>(g, "theta", resDim);
    if(theta && resDim.size() == 2 && resDim[0] == 2 && resDim[1] == 1)
    {
        std::cout << timestamp << "YAML theta..." << std::endl;
        config["theta"] = YAML::Load("[]");
        config["theta"].push_back(theta[0]);
        config["theta"].push_back(theta[1]);
    }
    resDim.clear();

    doubleArr res = hdf5util::getArray<double>(g, "resolution", resDim);
    if(res && resDim.size() == 2 && resDim[0] == 2 && resDim[1] == 1)
    {
        std::cout << timestamp << "YAML resolution..." << std::endl;
        config["v_res"] = theta[0];
        config["h_res"] = theta[1];
    }
    resDim.clear();

    vector<size_t> v = hdf5util::getDimensions<float>(g, "points");
    if(v.size() == 2)
    {
        config["num_points"] = v[0];
    }

    node["config"] = config;
    return node;
}

YAML::Node HDF5MetaDescriptionV2::scanPosition(const HighFive::Group &g) const 
{
    YAML::Node node;

    // Get GPS information
    std::vector<size_t> dim;
    doubleArr gps = hdf5util::getArray<double>(g, "gpsPosition", dim);
    if(gps && dim.size() == 2 && dim[0] == 3 && dim[1] == 1)
    {
        std::cout << timestamp << "YAML GPS..." << std::endl;
        node["latitude"] = gps[0];
        node["longitude"] = gps[1];
        node["altitude"] = gps[2];
    }
    dim.clear();

    // Get timestamp
    doubleArr ts = hdf5util::getArray<double>(g, "gpsPosition", dim);
    if(gps && dim.size() == 2 && dim[0] == 1 && dim[1] == 1)
    {
        std::cout << timestamp << "YAML timestamp..." << std::endl;
        node["timestamp"] = ts[0];
    }
    dim.clear();

    // Get pose estimation and registration
    boost::optional<Transformd> poseEstimate = 
        hdf5util::getMatrix<Transformd>(g, "poseEstimation");
    if(poseEstimate)
    {
        node["pose_estimate"] = *poseEstimate;
    }

    boost::optional<Transformd> registration = 
        hdf5util::getMatrix<Transformd>(g, "registration");
    if(poseEstimate)
    {
        node["registration"] = *registration;
    }

    return node;
}

YAML::Node HDF5MetaDescriptionV2::scanProject(const HighFive::Group &g) const 
{
    std::cout << timestamp << "HDF5MetaDescriptionV2::scanProject() not implemented..." << std::endl;
    YAML::Node node;
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


} // namespace lvr2