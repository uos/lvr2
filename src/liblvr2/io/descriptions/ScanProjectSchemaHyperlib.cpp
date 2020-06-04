#include <sstream> 
#include <iomanip>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaHyperlib.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/yaml/Scan.hpp"
#include "lvr2/io/yaml/ScanImage.hpp"
#include "lvr2/io/yaml/ScanPosition.hpp"
#include "lvr2/io/yaml/MatrixIO.hpp"
#include "lvr2/io/yaml/ScanProject.hpp"
#include "lvr2/io/yaml/ScanCamera.hpp"

namespace lvr2
{

Description ScanProjectSchemaHyperlib::scanProject() const
{
    Description d;
    d.groupName = boost::none;           // All scan related data is stored in the "raw" group
    d.dataSetName = boost::none;    // No dataset name for project root
    d.metaName = "meta.yaml";
    d.metaData = boost::none;

    boost::filesystem::path metaPath(*d.metaName);
    try
    {
        d.metaData = YAML::LoadFile(metaPath.string());
    }
    catch(const YAML::BadFile& e)
    {
        // Nothing to do, meta node will contain default values
        YAML::Node node;
        node = ScanProject();
        d.metaData = node;
    }
    
    return d;
}

Description ScanProjectSchemaHyperlib::position(const size_t &scanPosNo) const
{
    Description d; 
   
    // Save scan file name
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanPosNo;
    
    d.dataSetName = boost::none;
    d.metaName = "meta.yaml";
    d.metaData = boost::none;

    // Load meta data
    boost::filesystem::path positionPath(sstr.str());
    boost::filesystem::path metaPath(*d.metaName);

    d.groupName = (positionPath).string();
    
    try
    {
        d.metaData = YAML::LoadFile( (positionPath / metaPath).string());
    }
    catch(YAML::BadFile& e)
    {
         // Nothing to do, meta node will contail default values
        YAML::Node node;
        node = ScanPosition();
        d.metaData = node;
    }
    
    return d;
}

Description ScanProjectSchemaHyperlib::scan(const size_t &scanPosNo, const size_t &scanNo) const
{
    // Get information about scan the associated scan position
    Description d = position(scanPosNo);   
    return scan(*d.groupName, scanNo);
}

Description ScanProjectSchemaHyperlib::scan(const std::string &scanPositionPath, const size_t &scanNo) const
{

    Description d;
    boost::filesystem::path groupPath(scanPositionPath);
    boost::filesystem::path scansPath("scans");
    boost::filesystem::path dataPath("data");
    boost::filesystem::path totalGroupPath = groupPath / scansPath / dataPath;
    d.groupName = totalGroupPath.string();

    // Create dataset path
    std::stringstream sstr;
    sstr << "scan" << std::setfill('0') << std::setw(8) << scanNo;
    d.dataSetName = sstr.str() + std::string(".ply");

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
        node = Scan();
        d.metaData = node;
    }
   
    d.metaName = metaPath.string();
    d.groupName = totalGroupPath.string();
    return d;
}

Description ScanProjectSchemaHyperlib::scanCamera(const size_t &scanPositionNo, const size_t &camNo) const
{
    Description g = position(scanPositionNo);
    return scanCamera(*g.groupName, camNo);
}

Description ScanProjectSchemaHyperlib::scanCamera(const std::string &scanPositionPath, const size_t &camNo) const
{
    Description d;
   
    // Construct group path
    std::stringstream sstr;
    sstr << "cam_" << camNo;

    boost::filesystem::path groupPath(scanPositionPath);
    boost::filesystem::path camPath(sstr.str());
    d.groupName = (groupPath / camPath).string();

    // No data set information for camera position
    d.dataSetName = boost::none; 
    d.metaName = "meta.yaml";

    boost::filesystem::path metaPath(*d.metaName);

    // Load camera information from yaml
    d.metaData = boost::none;
    try
    {
         d.metaData = YAML::LoadFile( (groupPath / camPath / metaPath).string());
    }
    catch(const YAML::BadFile& e)
    {
         // Nothing to do, meta node will contail default values
        YAML::Node node;
        node = ScanCamera();
        d.metaData = node;
    }
    
   

    return d;
}

Description ScanProjectSchemaHyperlib::scanImage(
    const size_t &scanPosNo, const size_t &scanNo,
    const size_t &scanCameraNo, const size_t &scanImageNo) const
{
    // Scan images are not supported
    Description d_cam = scanCamera(scanPosNo, scanCameraNo);
    return scanImage(*d_cam.groupName, scanImageNo);
}

Description ScanProjectSchemaHyperlib::scanImage(
    const std::string &scanImagePath, const size_t &scanImageNo) const
{
    Description d;

    boost::filesystem::path siPath(scanImagePath);
    boost::filesystem::path dPath("data");

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanImageNo;
    
    std::string imgName(sstr.str() + std::string(".png"));
    std::string yamlName(sstr.str() + std::string(".yaml"));

    boost::filesystem::path metaPath(yamlName);

    d.groupName = (siPath / dPath).string();
    d.dataSetName = imgName;
    d.metaName = yamlName;
    d.metaData = boost::none;

    try
    {
        d.metaData = YAML::LoadFile((siPath / dPath / metaPath).string());
    }
    catch(YAML::BadFile& e)
    {
        // Nothing to do, meta node will contail default values
        YAML::Node node;
        node = ScanImage();
        d.metaData = node;
    }
   

    return d; 
}

} // namespace lvr2