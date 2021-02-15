#include <sstream> 
#include <iomanip>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/yaml.hpp"

#include "lvr2/io/descriptions/ScanProjectSchemaHDF5.hpp"

namespace lvr2
{

Description ScanProjectSchemaHDF5::scanProject() const
{
    Description d;
    d.groupName = "raw";
    d.metaName = "";
    return d;
}

Description ScanProjectSchemaHDF5::position(const size_t &scanPosNo) const
{
    Description d_parent = scanProject();

    Description d;
    
    // Save scan file name
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanPosNo;
    
    d.metaName = "";

    // Load meta data
    boost::filesystem::path positionPath(sstr.str());

    // append positionPath to parent path if necessary
    if(d_parent.groupName)
    {
        positionPath = boost::filesystem::path(*d_parent.groupName) / positionPath;
    }

    boost::filesystem::path metaPath(*d.metaName);
    d.groupName = (positionPath).string();
    
    return d;
}

Description ScanProjectSchemaHDF5::lidar(
    const Description& d_parent, 
    const size_t &lidarNo) const
{
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << lidarNo;

    Description d;
    d.groupName = "lidar_" + sstr.str();
    d.metaName = "";

    if(d_parent.groupName)
    {
        d.groupName = *d_parent.groupName + "/" +  *d.groupName;
    }

    return d;
}

Description ScanProjectSchemaHDF5::camera(
    const Description& d_parent,
    const size_t &camNo) const
{
    Description d;

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << camNo;

    d.groupName = "cam_" + sstr.str();
    d.metaName = "";

    if(d_parent.groupName)
    {
        d.groupName = *d_parent.groupName + "/" + *d.groupName;
    }

    return d;
}


Description ScanProjectSchemaHDF5::scan(
    const Description& d_parent,
    const size_t &scanNo) const
{
    Description d;

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanNo;

    d.groupName = sstr.str();
    d.metaName = "";

    if(d_parent.groupName)
    {
        d.groupName = *d_parent.groupName + "/" + *d.groupName;
    }

    return d;
}

Description ScanProjectSchemaHDF5::channel(
    const Description& d_parent,
    const std::string& channel_name) const
{
    Description d;

    d.dataSetName = channel_name;
    d.metaName = channel_name;

    if(d_parent.groupName)
    {
        d.groupName = *d_parent.groupName;
    }

    return d;
}

Description ScanProjectSchemaHDF5::cameraImage(
    const Description& d_parent, 
    const size_t &cameraImageNo) const
{
    Description d;

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << cameraImageNo;
   
    d.groupName = "";
    d.metaName = sstr.str();
    d.dataSetName = sstr.str();

    if(d_parent.groupName)
    {
        d.groupName = *d_parent.groupName + "/" + *d.groupName;
    }

    return d; 
}

Description ScanProjectSchemaHDF5::hyperspectralCamera(
    const Description& d_parent,
    const size_t &camNo) const
{
    Description d;

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << camNo;

    d.groupName = "hypercam_" + sstr.str();
    d.metaName = "";

    if(d_parent.groupName)
    {
        d.groupName = *d_parent.groupName + "/" + *d.groupName;
    }

    return d;
}

Description ScanProjectSchemaHDF5::hyperspectralPanorama(
    const Description& d_parent,
    const size_t& panoNo) const
{
    Description d;

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << panoNo;

    d.groupName = sstr.str();
    d.metaName = "";

    if(d_parent.groupName)
    {
        d.groupName = *d_parent.groupName + "/" + *d.groupName;
    }

    return d;
}

Description ScanProjectSchemaHDF5::hyperspectralPanoramaChannel(
    const Description& d_parent,
    const size_t& channelNo
) const
{
    Description d;

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << channelNo;
   
    d.groupName = "";
    d.metaName = sstr.str();
    d.dataSetName = sstr.str();

    if(d_parent.groupName)
    {
        d.groupName = *d_parent.groupName + "/" + *d.groupName;
    }

    return d;
}

} // namespace lvr2
