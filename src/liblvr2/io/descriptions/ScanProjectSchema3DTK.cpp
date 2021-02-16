#include <sstream> 
#include <iomanip>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/yaml.hpp"

#include "lvr2/io/descriptions/ScanProjectSchema3DTK.hpp"

namespace lvr2
{

Description ScanProjectSchema3DTK::scanProject() const
{
    Description d;

    d.dataRoot = "slam6d";

    d.metaRoot = d.dataRoot;
    // d.meta = "meta.slam6d";

    return d;
}

Description ScanProjectSchema3DTK::position(
    const size_t &scanPosNo) const
{
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(3) << scanPosNo;

    Description dp = scanProject();
    Description d;
    d.dataRoot = dp.dataRoot;
    d.metaRoot = d.dataRoot;

    // write slam6d at scanposition level
    d.meta = "scan" + sstr.str() + ".slam6d";
    
    return d;
}

Description ScanProjectSchema3DTK::lidar(
    const size_t& scanPosNo, 
    const size_t& lidarNo) const
{
    Description d;

    if(lidarNo == 0)
    {
        Description dp = position(scanPosNo);
        d.dataRoot = dp.dataRoot;
    }

    return d;
}

Description ScanProjectSchema3DTK::camera(
    const size_t& scanPosNo,
    const size_t& camNo) const
{
    Description d;
    
    return d;
}


Description ScanProjectSchema3DTK::scan(
    const size_t& scanPosNo,
    const size_t& lidarNo,
    const size_t& scanNo) const
{
    Description d;

    if(lidarNo == 0 && scanNo == 0)
    {
        // extract first scan of first LIDAR
        std::stringstream sstr;
        sstr << std::setfill('0') << std::setw(3) << scanPosNo;
        Description dp = lidar(scanPosNo, lidarNo);
        d.dataRoot = *dp.dataRoot;
        d.data = "scan" + sstr.str() + ".3d";
    }
    
    return d;
}

Description ScanProjectSchema3DTK::scanChannel(
    const size_t& scanPosNo,
    const size_t& lidarNo,
    const size_t& scanNo,
    const std::string& channelName) const
{
    Description d;

    return d;
}


Description ScanProjectSchema3DTK::cameraImage(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& cameraImageNo) const
{
    Description d;

    return d;
}

Description ScanProjectSchema3DTK::hyperspectralCamera(
    const size_t& scanPosNo,
    const size_t& camNo) const
{
    Description d;

    return d;
}

Description ScanProjectSchema3DTK::hyperspectralPanorama(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& panoNo) const
{
    Description d;

    return d;
}

Description ScanProjectSchema3DTK::hyperspectralPanoramaChannel(
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
