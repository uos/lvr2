#include <sstream> 
#include <iomanip>

#include "lvr2/io/descriptions/ScanProjectSchemaHDF5V2.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/yaml/MatrixIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

#include <boost/filesystem.hpp>

namespace lvr2
{

Description ScanProjectSchemaHDF5V2::scanProject() const
{
    Description d;
    d.groupName = "";           // All data is saved in the root dir
    d.dataSetName = boost::none;    // No dataset name for project root
    d.metaData = boost::none;       // No metadata for project 
    return d;
}
Description ScanProjectSchemaHDF5V2::position(const size_t &scanPosNo) const
{
    Description d; 

    // Group name
    std::stringstream scan_stream;
    scan_stream << "/raw/" << std::setfill('0') << std::setw(8) << scanPosNo;
    d.groupName = scan_stream.str();

    // No dataset name 
    d.dataSetName = boost::none;

    // No meta data, currently handled by meta data description
    d.metaData = boost::none;
    d.metaName = boost::none;

    return d;
}

Description ScanProjectSchemaHDF5V2::scan(const size_t &scanPosNo, const size_t &scanNo) const 
{
    // Group name
    std::stringstream group_stream;
    group_stream << "/raw/" << std::setfill('0') << std::setw(8) << scanPosNo << "/scans/data/" << std::setfill('0') << std::setw(8) << scanNo;
  
    return scan(group_stream.str(), scanNo);
}

Description ScanProjectSchemaHDF5V2::scan(const std::string& scanPositionPath, const size_t &scanNo) const
{
    Description d; 
    std::cout << "DEBUG: " << scanPositionPath << std::endl;
    d.groupName = scanPositionPath;

    // Scan name is always points in the 
    // respective HDF5 group
    d.dataSetName = "points";

    return d;
}

Description ScanProjectSchemaHDF5V2::scanCamera(const size_t &scanPositionNo, const size_t &camNo) const
{
    // Scan camera is not supported
    Description d;
    d.groupName = boost::none;
    d.dataSetName = boost::none;
    d.metaData = boost::none;
    d.metaName = boost::none;
    return d;
}

Description ScanProjectSchemaHDF5V2::scanCamera(const std::string &scanPositionPath, const size_t &camNo) const
{
    // Scan camera is not supported
    Description d;
    d.groupName = boost::none;
    d.dataSetName = boost::none;
    d.metaData = boost::none;
    d.metaName = boost::none;
    return d;
}

Description ScanProjectSchemaHDF5V2::scanImage(
    const size_t &scanPosNo, const size_t &scanNo,
    const size_t &scanCameraNo, const size_t &scanImageNo) const
{
    // Scan images are not supported
    Description d;
    d.groupName = boost::none;
    d.dataSetName = boost::none;
    d.metaData = boost::none;
    d.metaName = boost::none;
    return d;
}

Description ScanProjectSchemaHDF5V2::scanImage(
    const std::string &scanImagePath, const size_t &scanImageNo) const
{
    // Scan images are not supported
    Description d;
    d.groupName = boost::none;
    d.dataSetName = boost::none;
    d.metaData = boost::none;
    d.metaName = boost::none;
    return d; 
}

} // namespace lvr2