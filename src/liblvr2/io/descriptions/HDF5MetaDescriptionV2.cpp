#include "lvr2/io/descriptions/HDF5MetaDescriptionV2.hpp"
#include "lvr2/io/yaml/MetaNodeDescriptions.hpp"
#include "lvr2/io/hdf5/Hdf5Util.hpp"


namespace lvr2
{

void HDF5MetaDescriptionV2::saveHyperspectralCamera(
    const HighFive::Group &g,
    const YAML::Node& n) const
{

}

void HDF5MetaDescriptionV2::saveHyperspectralPanoramaChannel(
    const HighFive::Group &g,
    const YAML::Node &n) const
{

}

void HDF5MetaDescriptionV2::saveScan(
    const HighFive::Group &g,
    const YAML::Node &n) const
{

}

void HDF5MetaDescriptionV2::saveScanPosition(
    const HighFive::Group &g,
    const YAML::Node &n) const
{

}

void HDF5MetaDescriptionV2::saveScanProject(
    const HighFive::Group &g,
    const YAML::Node &n) const 
{

}

void HDF5MetaDescriptionV2::saveScanCamera(
    const HighFive::Group &g,
    const YAML::Node& n) const
{

}

void HDF5MetaDescriptionV2::saveScanImage(
    const HighFive::Group &g,
    const YAML::Node &n) const
{

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