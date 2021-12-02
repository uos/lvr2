#include <sstream> 
#include <iomanip>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/scanio/yaml/YAML.hpp"
#include "lvr2/io/scanio/ScanProjectSchemaHDF5.hpp"

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
    const std::vector<size_t>& cameraImageNos) const
{
    if(cameraImageNos.size() == 0)
    {
        std::cout << "ERROR: cameraImageNos size = 0" << std::endl;   
    }

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << cameraImageNos[0];

    for(size_t i=1; i<cameraImageNos.size(); i++)
    {
        sstr << "/" << std::setfill('0') << std::setw(8) << cameraImageNos[i];
    }

    Description dp = camera(scanPosNo, camNo);
   
    Description d;
    d.dataRoot = dp.dataRoot;
    d.data = sstr.str();
    d.metaRoot = dp.metaRoot;
    d.meta = sstr.str();

    return d; 
}

Description ScanProjectSchemaHDF5::cameraImageGroup(
    const size_t& scanPosNo,
    const size_t& camNo, 
    const std::vector<size_t>& cameraImageGroupNos) const
{
    return cameraImage(scanPosNo, camNo, cameraImageGroupNos);
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
    d.meta = sstr.str();
    d.data = sstr.str();

    return d;
}

} // namespace lvr2
