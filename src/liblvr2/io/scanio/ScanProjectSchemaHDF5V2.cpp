#include <sstream> 
#include <iomanip>

#include "lvr2/io/scanio/ScanProjectSchemaHDF5V2.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/yaml/MatrixIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

#include <boost/filesystem.hpp>

namespace lvr2
{

Description ScanProjectSchemaHDF5V2::scanProject() const
{
    Description d;
    d.groupName = "raw";           // All data is saved in the root dir
    d.dataSetName = boost::none;    // No dataset name for project root
    d.metaData = boost::none;       // No metadata for project 
    d.metaName = "meta";
    return d;
}
Description ScanProjectSchemaHDF5V2::position(const size_t &scanPosNo) const
{
    Description d_parent = scanProject();

    Description d;
    
    // Save scan file name
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanPosNo;
    
    d.dataSetName = boost::none;
    d.metaName = "meta";
    d.metaData = boost::none;

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

Description ScanProjectSchemaHDF5V2::scan(const size_t &scanPosNo, const size_t &scanNo) const 
{
    // Get information about scan the associated scan position
    Description d_parent = position(scanPosNo);
    return scan(*d_parent.groupName, scanNo);
}

Description ScanProjectSchemaHDF5V2::scan(const std::string& scanPositionPath, const size_t &scanNo) const
{
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanNo;

    Description d;
    boost::filesystem::path scansPath = boost::filesystem::path(scanPositionPath) / "scans";
    boost::filesystem::path scanPath = scansPath / sstr.str();

    d.groupName = scanPath.string();
    d.dataSetName = "data";
    d.metaName = "meta";

    return d;
}

Description ScanProjectSchemaHDF5V2::scanCamera(const size_t &scanPositionNo, const size_t &camNo) const
{
    // Group name
    Description d_parent = position(scanPositionNo);
    return scanCamera(*d_parent.groupName, camNo);
}

Description ScanProjectSchemaHDF5V2::scanCamera(const std::string &scanPositionPath, const size_t &camNo) const
{
    Description d;
   
    // Construct group path
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << camNo;

    boost::filesystem::path positionPath(scanPositionPath);
    boost::filesystem::path camerasPath("cameras");
    boost::filesystem::path groupPath = scanPositionPath / camerasPath;

    boost::filesystem::path camPath(sstr.str());
    d.groupName = (groupPath / camPath).string();

    // No data set information for camera position
    d.dataSetName = "data"; 
    d.metaName = "meta";
    d.metaData = boost::none;

    return d;
}

Description ScanProjectSchemaHDF5V2::scanImage(
    const size_t &scanPosNo, 
    const size_t &scanCameraNo, 
    const size_t &scanImageNo) const
{
    // Scan images are not supported
    Description d_cam = scanCamera(scanPosNo, scanCameraNo);
    return scanImage((*d_cam.groupName + "/" + *d_cam.dataSetName) , scanImageNo);
}

Description ScanProjectSchemaHDF5V2::scanImage(
    const std::string &scanImagePath, const size_t &scanImageNo) const
{
    Description d;

    // data/images/%8d
    // data/metas/%8d

    boost::filesystem::path siPath(scanImagePath);
    boost::filesystem::path iPath("images");
    boost::filesystem::path mPath("meta");

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanImageNo;
    
    std::string imgName(sstr.str());
    std::string metaName(sstr.str());

    d.groupName = (siPath).string();
    d.dataSetName = (iPath / imgName).string();
    d.metaName = (mPath / metaName).string();
    d.metaData = boost::none;

    return d; 
}

} // namespace lvr2