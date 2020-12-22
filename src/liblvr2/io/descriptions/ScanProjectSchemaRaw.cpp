#include <sstream> 
#include <iomanip>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaRaw.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/yaml/Scan.hpp"
#include "lvr2/io/yaml/Waveform.hpp"
#include "lvr2/io/yaml/Label.hpp"
#include "lvr2/io/yaml/ScanImage.hpp"
#include "lvr2/io/yaml/ScanPosition.hpp"
#include "lvr2/io/yaml/MatrixIO.hpp"
#include "lvr2/io/yaml/ScanProject.hpp"
#include "lvr2/io/yaml/ScanCamera.hpp"

#include <boost/optional/optional_io.hpp>


namespace lvr2
{

Description ScanProjectSchemaRaw::scanProject() const
{
    Description d;
    d.groupName = "raw";           // All scan related data is stored in the "raw" group
    d.dataSetName = boost::none;    // No dataset name for project root
    d.metaName = "meta.yaml";
    d.metaData = boost::none;
    
    return d;
}

Description ScanProjectSchemaRaw::position(const size_t &scanPosNo) const
{
    Description d_parent = scanProject();

    Description d;
    
    // Save scan file name
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanPosNo;
    
    d.dataSetName = boost::none;
    d.metaName = "meta.yaml";
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

Description ScanProjectSchemaRaw::scan(const size_t &scanPosNo, const size_t &scanNo) const
{
    // Get information about scan the associated scan position
    Description d = position(scanPosNo);
    return scan(*d.groupName, scanNo);
}

Description ScanProjectSchemaRaw::scan(const std::string &scanPositionPath, const size_t &scanNo) const
{
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanNo;

    Description d;
    boost::filesystem::path scansPath = boost::filesystem::path(scanPositionPath) / "scans";
    boost::filesystem::path scanPath = scansPath / sstr.str();

    d.groupName = scanPath.string();
    d.dataSetName = "channels";
    d.metaName = "meta.yaml";

    return d;
}

Description ScanProjectSchemaRaw::waveform(const size_t &scanPosNo, const size_t &scanNo) const
{
    // Get information about scan the associated scan position
    Description d = position(scanPosNo);   
    return waveform(*d.groupName, scanNo);
}

Description ScanProjectSchemaRaw::waveform(const std::string &scanPositionPath, const size_t &scanNo) const
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

Description ScanProjectSchemaRaw::scanCamera(const size_t &scanPositionNo, const size_t &camNo) const
{
    Description g = position(scanPositionNo);
    return scanCamera(*g.groupName, camNo);
}

Description ScanProjectSchemaRaw::scanCamera(const std::string &scanPositionPath, const size_t &camNo) const
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

    d.metaData = boost::none;

    return d;
}

Description ScanProjectSchemaRaw::scanImage(
    const size_t &scanPosNo, const size_t &scanCameraNo, const size_t &scanImageNo) const
{
    // Scan images are not supported
    Description d_cam = scanCamera(scanPosNo, scanCameraNo);
    return scanImage(*d_cam.groupName, scanImageNo);
}

Description ScanProjectSchemaRaw::scanImage(
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
        d.metaData = YAML::LoadFile((m_rootPath / siPath / dPath / metaPath).string());
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
