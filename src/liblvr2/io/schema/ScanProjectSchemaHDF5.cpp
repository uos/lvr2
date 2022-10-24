#include <sstream> 
#include <iomanip>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/YAML.hpp"
#include "lvr2/io/schema/ScanProjectSchemaHDF5.hpp"

namespace lvr2
{

Description ScanProjectSchemaHDF5::scanProject() const
{
    Description d;
    d.dataRoot = "raw";
    d.metaRoot = d.dataRoot;
    d.meta = "";
    return d;
}

Description ScanProjectSchemaHDF5::position(
    const size_t& scanPosNo) const
{
    Description dp = scanProject();

    Description d;
    // Save scan file name
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanPosNo;
    
    d.dataRoot = *dp.dataRoot + "/" + sstr.str();
    d.metaRoot = d.dataRoot;
    d.meta = "";

    
    return d;
}

Description ScanProjectSchemaHDF5::lidar(
    const size_t& scanPosNo, 
    const size_t& lidarNo) const
{
    Description dp = position(scanPosNo);

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << lidarNo;

    Description d;
    d.dataRoot = *dp.dataRoot + "/lidar_" + sstr.str();
    d.metaRoot = d.dataRoot;
    d.meta = "";

    return d;
}

Description ScanProjectSchemaHDF5::cameraImageGroup(
    const size_t &scanPosNo,
    const size_t &camNo,
    const size_t &groupNo) const
{
    Description dp = camera(scanPosNo, camNo);
    
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << groupNo;

    Description d;
    d.dataRoot = *dp.dataRoot + "/" + sstr.str();
    d.metaRoot = d.dataRoot;
    d.meta = "";

    return d;
}

Description ScanProjectSchemaHDF5::camera(
    const size_t& scanPosNo,
    const size_t& camNo) const
{
    Description dp = position(scanPosNo);

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << camNo;

    Description d;
    d.dataRoot = *dp.dataRoot + "/cam_" + sstr.str();
    d.metaRoot = d.dataRoot;
    d.meta = "";


    return d;
}


Description ScanProjectSchemaHDF5::scan(
    const size_t& scanPosNo,
    const size_t& lidarNo,
    const size_t& scanNo) const
{
    Description dp = lidar(scanPosNo, lidarNo);

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanNo;

    Description d;

    d.dataRoot = *dp.dataRoot + "/" + sstr.str();
    d.metaRoot = d.dataRoot;
    d.meta = "";

    return d;
}

Description ScanProjectSchemaHDF5::scanChannel(
    const size_t& scanPosNo,
    const size_t& lidarNo,
    const size_t& scanNo,
    const std::string& channel_name) const
{
    Description dp = scan(scanPosNo, lidarNo, scanNo);
    
    Description d;

    d.dataRoot = dp.dataRoot;
    d.data = channel_name;
    d.metaRoot = dp.metaRoot;
    d.meta = channel_name;

    return d;
}

Description ScanProjectSchemaHDF5::cameraImage(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& groupNo, 
    const size_t& imgNo) const
{
    Description dp = cameraImageGroup(scanPosNo, camNo, groupNo);

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << imgNo;

    Description d;
    d.dataRoot = *dp.dataRoot;
    d.metaRoot = d.dataRoot;
    d.data = sstr.str();
    d.meta = sstr.str();

    return d; 
}

Description ScanProjectSchemaHDF5::hyperspectralCamera(
    const size_t& scanPosNo,
    const size_t& camNo) const
{
    Description dp = position(scanPosNo);

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << camNo;

    Description d;
    d.dataRoot = *dp.dataRoot + "/" + "hypercam_" + sstr.str();
    d.metaRoot = d.dataRoot;
    d.meta = "";

    return d;
}

Description ScanProjectSchemaHDF5::hyperspectralPanorama(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& panoNo) const
{
    Description dp = hyperspectralCamera(scanPosNo, camNo);

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << panoNo;
    Description d;
    d.dataRoot = *dp.dataRoot + "/" + sstr.str();
    d.metaRoot = d.dataRoot;
    d.meta = "";

    return d;
}

Description ScanProjectSchemaHDF5::hyperspectralPanoramaPreview(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& panoNo) const
{
    Description dp = hyperspectralPanorama(scanPosNo, camNo, panoNo);
    Description d;

    d.dataRoot = *dp.dataRoot;
    d.metaRoot = d.dataRoot;
    d.data = "preview";

    return d;
}

Description ScanProjectSchemaHDF5::hyperspectralPanoramaChannel(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& panoNo,
    const size_t& channelNo) const
{
    Description dp = hyperspectralPanorama(scanPosNo, camNo, panoNo);

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << channelNo;
   
    Description d;
    d.dataRoot = dp.dataRoot;
    d.metaRoot = dp.metaRoot;

    d.meta = "frames";
    d.data = "frames";

    return d;
}

} // namespace lvr2
