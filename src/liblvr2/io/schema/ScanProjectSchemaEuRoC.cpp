#include <sstream> 
#include <iomanip>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/YAML.hpp"
#include "lvr2/io/schema/ScanProjectSchemaEuRoC.hpp"

namespace lvr2
{

Description ScanProjectSchemaEuRoC::scanProject() const
{
    Description d;
    d.dataRoot = "";
    d.metaRoot = d.dataRoot;
    return d;
}

Description ScanProjectSchemaEuRoC::position(
    const size_t &scanPosNo) const
{
    Description d;

    if(scanPosNo > 0)
    {
        return d;
    }


    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanPosNo;

    Description dp = scanProject();
    d.dataRoot = *dp.dataRoot;
    d.metaRoot = d.dataRoot;
    // d.meta = "meta.yaml";
    
    return d;
}

Description ScanProjectSchemaEuRoC::lidar(
    const size_t& scanPosNo, 
    const size_t& lidarNo) const
{

    std::stringstream sstr;
    sstr << lidarNo;

    Description dp = position(scanPosNo);

    Description d;
    d.dataRoot = *dp.dataRoot + "/pointcloud" + sstr.str();

    d.metaRoot = d.dataRoot;
    d.meta = "sensor.yaml";

    return d;
}

Description ScanProjectSchemaEuRoC::camera(
    const size_t& scanPosNo,
    const size_t& camNo) const
{
    std::stringstream sstr;
    sstr << camNo;

    Description dp = position(scanPosNo);

    Description d;
    d.dataRoot = *dp.dataRoot + "/cam" + sstr.str();

    d.metaRoot = d.dataRoot;
    d.meta = "sensor.yaml";
    
    return d;
}


Description ScanProjectSchemaEuRoC::scan(
    const size_t& scanPosNo,
    const size_t& lidarNo,
    const size_t& scanNo) const
{
    Description dp = lidar(scanPosNo, lidarNo);

    Description d;
    if(scanNo > 0)
    {
        return d;
    }

    d.dataRoot = *dp.dataRoot;
    d.data = "data.ply";
    d.metaRoot = d.dataRoot;
    
    return d;
}

Description ScanProjectSchemaEuRoC::scanChannel(
    const size_t& scanPosNo,
    const size_t& lidarNo,
    const size_t& scanNo,
    const std::string& channelName) const
{
    Description d;
    return d;
}

Description ScanProjectSchemaEuRoC::cameraImage(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& groupNo,
    const size_t& imgNo) const
{
    Description d;
    Description dp = camera(scanPosNo, camNo);

    // wtf
    size_t start = 1403715273262142976;
    size_t inc_odd = 50000128;
    // size_t inc_even = 49999872;
    size_t inc_dbl = 100000000;
 
    size_t curr = start + inc_dbl * (imgNo / 2);
    if(imgNo % 2)
    {
       // odd
       curr +=  inc_odd;
    }

    std::stringstream sstr;
    sstr << curr << ".png";

    d.dataRoot = *dp.dataRoot + "/data";
    d.data = sstr.str();
    d.metaRoot = d.dataRoot;

    // std::cout << "Loading " << imgNo << ", " << *d.data << std::endl;

    return d;
}

Description ScanProjectSchemaEuRoC::cameraImageGroup(
    const size_t &scanPosNo,
    const size_t &camNo,
    const size_t &GroupNo) const
{
    Description d;
    return d;
}

Description ScanProjectSchemaEuRoC::hyperspectralCamera(
    const size_t& scanPosNo,
    const size_t& camNo) const
{
    Description d;
    return d;
}

Description ScanProjectSchemaEuRoC::hyperspectralPanorama(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& panoNo) const
{
    Description d;
    return d;
}

Description ScanProjectSchemaEuRoC::hyperspectralPanoramaPreview(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& panoNo) const
{
    Description d;
    return d;
}

Description ScanProjectSchemaEuRoC::hyperspectralPanoramaChannel(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& panoNo,
    const size_t& channelNo
) const
{
    Description d;
    return d;
}

} // namespace lvr2
