#include <sstream> 
#include <iomanip>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/yaml.hpp"

#include "lvr2/io/descriptions/ScanProjectSchemaSlam6D.hpp"

namespace lvr2
{

Description ScanProjectSchemaSlam6D::scanProject() const
{
    Description d;

    d.dataRoot = "slam6d";

    d.metaRoot = d.dataRoot;
    // d.meta = "meta.slam6d";

    return d;
}

Description ScanProjectSchemaSlam6D::position(
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

Description ScanProjectSchemaSlam6D::lidar(
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

Description ScanProjectSchemaSlam6D::camera(
    const size_t& scanPosNo,
    const size_t& camNo) const
{
    Description d;
    
    return d;
}


Description ScanProjectSchemaSlam6D::scan(
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

Description ScanProjectSchemaSlam6D::scanChannel(
    const size_t& scanPosNo,
    const size_t& lidarNo,
    const size_t& scanNo,
    const std::string& channelName) const
{
    Description d;

    return d;
}


Description ScanProjectSchemaSlam6D::cameraImage(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& cameraImageNo) const
{
    Description d;

    return d;
}

Description ScanProjectSchemaSlam6D::hyperspectralCamera(
    const size_t& scanPosNo,
    const size_t& camNo) const
{
    Description d;

    return d;
}

Description ScanProjectSchemaSlam6D::hyperspectralPanorama(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& panoNo) const
{
    Description d;

    return d;
}

Description ScanProjectSchemaSlam6D::hyperspectralPanoramaChannel(
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
