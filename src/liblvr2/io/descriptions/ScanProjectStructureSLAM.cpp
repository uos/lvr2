#include <sstream> 
#include <iomanip>

#include "lvr2/io/descriptions/ScanProjectStructureSLAM.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/yaml/MatrixIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

#include <boost/filesystem.hpp>

namespace lvr2
{

Description ScanProjectStructureSLAM::scanProject() const
{
    Description d;
    d.groupName = m_root;           // All data is saved in the root dir
    d.dataSetName = boost::none;    // No dataset name for project root
    d.metaData = boost::none;       // No metadata for project 
    return d;
}
Description ScanProjectStructureSLAM::position(const size_t &scanPosNo) const
{
    Description d; 
    
    // Scan does not support scan positions. So we only consider
    // scan position 0 valid
    if(scanPosNo == 0)
    {
        d.groupName = m_root;
    }
    else
    {
        d.groupName = boost::none;
    }

    d.dataSetName = boost::none;
    d.metaData = boost::none;
    d.metaName = boost::none;
    return d;
}

Description ScanProjectStructureSLAM::scan(const size_t &scanPosNo, const size_t &scanNo) const 
{
    Description d; 
    
    // Scan does not support scan positions. So we only consider
    // scan position 0 valid
    if(scanPosNo == 0)
    {
        d.groupName = m_root;
    }
    else
    {
        d.groupName = boost::none;
    }

    // Save scan file name
    std::stringstream scan_stream;
    scan_stream << "scan" << std::setfill('0') << std::setw(3) << scanPosNo;
    d.dataSetName = scan_stream.str() + ".3d";

    // Setup meta info -> Read frames file and pose file to setup valid 
    // meta description node
    boost::filesystem::path root_path(m_root);
    boost::filesystem::path pose_path(scan_stream.str() + ".pose");
    boost::filesystem::path frames_path(scan_stream.str() + ".frames");
    boost::filesystem::path scan_path(*d.dataSetName);
    
    Eigen::Matrix<double, 4, 4> poseEstimate = getTransformationFromPose<double>(root_path / pose_path);
    Eigen::Matrix<double, 4, 4> registration = getTransformationFromFrames<double>(root_path / frames_path);

    size_t num_pts = countPointsInFile(root_path / scan_path);

    YAML::Node node;
    node["sensor_type"] = lvr2::Scan::sensorType;   // Laser scanner

    node["start_time"] = 0.0;                       // Unknown
    node["end_time"] = 0.0;                         // Unknown

    node["pose_estimate"] = poseEstimate;           // Estimation from .pose          
    node["registration"] = registration;            // Registration from .frames

    YAML::Node config;
    config["theta"] = YAML::Load("[]");
    config["theta"].push_back(0);
    config["theta"].push_back(0);

    config["phi"] = YAML::Load("[]");
    config["phi"].push_back(0);
    config["phi"].push_back(0);

    config["v_res"] = 0.0;
    config["h_res"] = 0.0;

    config["num_points"] = num_pts;
    node["config"] = config;

    d.metaData = node;

    // Mark slam6d format
    d.metaName = scan_stream.str() + ".slam6d";
    return d;
}

Description ScanProjectStructureSLAM::scan(const std::string& scanPositionPath, const size_t &scanNo) const
{
    return scan(0, scanNo);
}

Description ScanProjectStructureSLAM::scanCamera(const size_t &scanPositionNo, const size_t &camNo) const
{
    // Scan camera is not supported
    Description d;
    d.groupName = boost::none;
    d.dataSetName = boost::none;
    d.metaData = boost::none;
    d.metaName = boost::none;
    return d;
}

Description ScanProjectStructureSLAM::scanCamera(const std::string &scanPositionPath, const size_t &camNo) const
{
    // Scan camera is not supported
    Description d;
    d.groupName = boost::none;
    d.dataSetName = boost::none;
    d.metaData = boost::none;
    d.metaName = boost::none;
    return d;
}

Description ScanProjectStructureSLAM::scanImage(
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

Description ScanProjectStructureSLAM::scanImage(
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