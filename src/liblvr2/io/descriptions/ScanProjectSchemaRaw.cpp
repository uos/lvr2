#include <sstream> 
#include <iomanip>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/yaml.hpp"

#include "lvr2/io/descriptions/ScanProjectSchemaRaw.hpp"

namespace lvr2
{

Description ScanProjectSchemaRaw::scanProject() const
{
    Description d;
    d.groupName = "raw";
    d.metaName = "meta.yaml";
    return d;
}

Description ScanProjectSchemaRaw::position(const size_t &scanPosNo) const
{
    Description d_parent = scanProject();

    Description d;
    
    // Save scan file name
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanPosNo;
    
    d.metaName = "meta.yaml";

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

Description ScanProjectSchemaRaw::lidar(
    const Description& d_parent, 
    const size_t &lidarNo) const
{
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << lidarNo;

    Description d;
    d.groupName = "lidar_" + sstr.str();
    d.metaName = "meta.yaml";
    
    if(d_parent.groupName)
    {
        d.groupName = *d_parent.groupName + "/" +  *d.groupName;
    }


    return d;
}

Description ScanProjectSchemaRaw::camera(
    const Description& d_parent,
    const size_t &camNo) const
{
    Description d;

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << camNo;

    d.groupName = "cam_" + sstr.str();
    d.metaName = "meta.yaml";

    if(d_parent.groupName)
    {
        d.groupName = *d_parent.groupName + "/" + *d.groupName;
    }

    return d;
}


Description ScanProjectSchemaRaw::scan(
    const Description& d_parent,
    const size_t &scanNo) const
{
    Description d;

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanNo;

    d.groupName = sstr.str();
    d.metaName = "meta.yaml";

    if(d_parent.groupName)
    {
        d.groupName = *d_parent.groupName + "/" + *d.groupName;
    }

    return d;
}

Description ScanProjectSchemaRaw::cameraImage(
    const Description& d_parent, 
    const size_t &cameraImageNo) const
{
    Description d;

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << cameraImageNo;
   
    d.groupName = sstr.str();
    d.metaName = "meta.yaml";
    d.dataSetName = "image.png";

    if(d_parent.groupName)
    {
        d.groupName = *d_parent.groupName + "/" + *d.groupName;
    }

    return d; 
}

} // namespace lvr2
