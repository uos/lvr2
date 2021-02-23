#include <sstream> 
#include <iomanip>

#include "lvr2/io/scanio/LabelScanProjectSchemaHDF5V2.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/yaml/MatrixIO.hpp"
#include "lvr2/io/yaml/Label.hpp"
#include "lvr2/io/yaml/Waveform.hpp"
#include "lvr2/types/ScanTypes.hpp"

#include <boost/filesystem.hpp>

namespace lvr2
{

Description LabelScanProjectSchemaHDF5V2::scanProject() const
{
    Description d;
    d.groupName = "";           // All data is saved in the root dir
    d.dataSetName = boost::none;    // No dataset name for project root
    d.metaData = boost::none;       // No metadata for project 
    return d;
}
Description LabelScanProjectSchemaHDF5V2::position(const size_t &scanPosNo) const
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

Description LabelScanProjectSchemaHDF5V2::scan(const size_t &scanPosNo, const size_t &scanNo) const 
{
    // Group name
    std::stringstream group_stream;
    group_stream << "/raw/" << std::setfill('0') << std::setw(8) << scanPosNo << "/scans/data/" << std::setfill('0') << std::setw(8) << scanNo;
  
    return scan(group_stream.str(), scanNo);
}

Description LabelScanProjectSchemaHDF5V2::scan(const std::string& scanPositionPath, const size_t &scanNo) const
{
    Description d; 
    std::cout << "DEBUG: " << scanPositionPath << std::endl;
    d.groupName = scanPositionPath;

    // Scan name is always points in the 
    // respective HDF5 group
    d.dataSetName = "points";

    return d;
}

Description LabelScanProjectSchemaHDF5V2::waveform(const size_t &scanPosNo, const size_t &scanNo) const
{
    // Get information about scan the associated scan position
    Description d = position(scanPosNo);   
    return waveform(*d.groupName, scanNo);
}

Description LabelScanProjectSchemaHDF5V2::waveform(const std::string &scanPositionPath, const size_t &scanNo) const
{

    Description d;
    boost::filesystem::path groupPath(scanPositionPath);
    boost::filesystem::path scansPath("scans");
    boost::filesystem::path waveformPath("waveform");
    boost::filesystem::path totalGroupPath = groupPath / scansPath / waveformPath;
    d.groupName = totalGroupPath.string();

    // Create dataset path
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanNo;
    d.dataSetName = sstr.str() + std::string(".lwf");

    // Load meta data for scan
    boost::filesystem::path metaPath(sstr.str() + ".yaml");
    d.metaData = boost::none;
    try
    {
        d.metaData = YAML::LoadFile((totalGroupPath / metaPath).string());
    }
    catch(YAML::BadFile& e)
    {
        // Nothing to do, meta node will contail default values
        YAML::Node node;
        node = Waveform();
        d.metaData = node;
    }
   
    d.metaName = metaPath.string();
    d.groupName = totalGroupPath.string();
    return d;
}
Description LabelScanProjectSchemaHDF5V2::scanCamera(const size_t &scanPositionNo, const size_t &camNo) const
{
    // Group name
    std::stringstream group_stream;
    group_stream << "/raw/" << std::setfill('0') << std::setw(8) << scanPositionNo << "/cam_" << std::setfill('0') << std::setw(2) << camNo;
  
    return scanCamera(group_stream.str(), camNo);
}

Description LabelScanProjectSchemaHDF5V2::scanCamera(const std::string &scanPositionPath, const size_t &camNo) const
{
    // Scan camera is not supported
    Description d;
    d.groupName = scanPositionPath;
    d.dataSetName = "camera";
    return d;
}

Description LabelScanProjectSchemaHDF5V2::scanImage(
    const size_t &scanPosNo,
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

Description LabelScanProjectSchemaHDF5V2::scanImage(
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

Description LabelScanProjectSchemaHDF5V2::labelInstance(const std::string& group, const std::string& className, const std::string &instanceName) const
{
    Description d;
    boost::filesystem::path groupPath(group);
    boost::filesystem::path pointcloudPath("pointcloud");
    boost::filesystem::path classPath(className);
    boost::filesystem::path instancePath(instanceName);
    boost::filesystem::path totalGroupPath = groupPath / pointcloudPath / classPath;

    // Create dataset path
    d.dataSetName = instanceName + std::string(".ids");

    // Load meta data for scan
    boost::filesystem::path metaPath = instanceName + std::string("meta.yaml");
    d.metaData = boost::none;
    try
    {
        d.metaData = YAML::LoadFile((totalGroupPath / metaPath).string());
    }
    catch(YAML::BadFile& e)
    {
        // Nothing to do, meta node will contail default values
        YAML::Node node;
        node = LabelInstance();
        d.metaData = node;
    }
    d.metaName = metaPath.string();
    d.groupName = totalGroupPath.string();
    return d;
}
} // namespace lvr2
