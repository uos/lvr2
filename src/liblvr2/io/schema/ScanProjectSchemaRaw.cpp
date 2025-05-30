#include <sstream> 
#include <iomanip>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/YAML.hpp"
#include "lvr2/io/schema/ScanProjectSchemaRaw.hpp"

namespace lvr2
{

Description ScanProjectSchemaRaw::scanProject() const
{
    Description d;

    d.dataRoot = "raw";

    d.metaRoot = d.dataRoot;
    d.meta = "meta.yaml";

    return d;
}

Description ScanProjectSchemaRaw::position(
    const size_t &scanPosNo) const
{
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanPosNo;

    Description dp = scanProject();
    Description d;
    d.dataRoot = *dp.dataRoot + "/" + sstr.str();
    
    d.metaRoot = d.dataRoot;
    d.meta = "meta.yaml";
    
    return d;
}

Description ScanProjectSchemaRaw::lidar(
    const size_t& scanPosNo, 
    const size_t& lidarNo) const
{
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << lidarNo;


    Description dp = position(scanPosNo);

    Description d;
    d.dataRoot = *dp.dataRoot + "/lidar_" + sstr.str();

    d.metaRoot = d.dataRoot;
    d.meta = "meta.yaml";

    return d;
}

Description ScanProjectSchemaRaw::camera(
    const size_t& scanPosNo,
    const size_t& camNo) const
{
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << camNo;

    Description dp = position(scanPosNo);

    Description d;
    d.dataRoot = *dp.dataRoot + "/cam_" + sstr.str();

    d.metaRoot = d.dataRoot;
    d.meta = "meta.yaml";
    
    return d;
}


Description ScanProjectSchemaRaw::scan(
    const size_t& scanPosNo,
    const size_t& lidarNo,
    const size_t& scanNo) const
{
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanNo;

    Description dp = lidar(scanPosNo, lidarNo);

    Description d;
    d.dataRoot = *dp.dataRoot + "/" + sstr.str();

    d.metaRoot = d.dataRoot;
    d.meta = "meta.yaml";
    
    return d;
}

Description ScanProjectSchemaRaw::scanChannel(
    const size_t& scanPosNo,
    const size_t& lidarNo,
    const size_t& scanNo,
    const std::string& channelName) const
{
    Description d;

    Description dp = scan(scanPosNo, lidarNo, scanNo);

    // if(channelName == "points" || channelName == "normals" || channelName == "colors")
    // {
    //     d.dataRoot = *dp.dataRoot + "/points.ply";
    // } else {
    //     d.dataRoot = *dp.dataRoot;
    // }

    d.dataRoot = *dp.dataRoot;

    d.data = channelName;
    d.metaRoot = *dp.dataRoot;
    d.meta = channelName + ".yaml";

    return d;
}

Description ScanProjectSchemaRaw::cameraImage(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& groupNo,
    const size_t& cameraImageNo) const
{

    std::stringstream image_sstr;
    image_sstr << std::setfill('0') << std::setw(8) << cameraImageNo;

    Description dp = cameraImageGroup(scanPosNo, camNo, groupNo);

    Description d;
    d.dataRoot = *dp.dataRoot;
    d.data = "image_" + image_sstr.str() + ".png";

    d.metaRoot = d.dataRoot;
    d.meta = "meta_" + image_sstr.str() + ".yaml";

    return d;
}

Description ScanProjectSchemaRaw::cameraImageGroup(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& groupNo) const
{

    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << groupNo;

    Description dp = camera(scanPosNo, camNo);

    Description d;
    d.dataRoot = *dp.dataRoot + "/" + sstr.str();
    d.data = "";

    d.metaRoot = d.dataRoot;
    d.meta = "";

    return d;
}


Description ScanProjectSchemaRaw::hyperspectralCamera(
    const size_t& scanPosNo,
    const size_t& camNo) const
{
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << camNo;

    Description dp = position(scanPosNo);

    Description d;
    d.dataRoot =  *dp.dataRoot + "/hypercam_" + sstr.str();

    d.metaRoot = d.dataRoot;
    d.meta = "meta.yaml";

    return d;
}

Description ScanProjectSchemaRaw::hyperspectralPanorama(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& panoNo) const
{
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << panoNo;

    Description dp = hyperspectralCamera(scanPosNo, camNo);

    Description d;
    d.dataRoot =  *dp.dataRoot + "/" + sstr.str();

    d.metaRoot = d.dataRoot;
    d.meta = "meta.yaml";

    return d;
}

Description ScanProjectSchemaRaw::hyperspectralPanoramaPreview(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& panoNo) const
{
    Description dp = hyperspectralPanorama(scanPosNo, camNo, panoNo);
    Description d;

    d.dataRoot = *dp.dataRoot;
    d.metaRoot = d.dataRoot;
    d.data = "preview.png";

    return d;
}

Description ScanProjectSchemaRaw::hyperspectralPanoramaChannel(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& panoNo,
    const size_t& channelNo
) const
{
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << channelNo;

    Description dp = hyperspectralPanorama(scanPosNo, camNo, panoNo);

    Description d;
    d.dataRoot = *dp.dataRoot + "/data";
    d.data = sstr.str() + ".png";
    
    d.metaRoot = d.dataRoot;
    d.meta = boost::none;
    
    return d;
}

Description ScanProjectSchemaRawPly::scan( 
        const size_t& scanPosNo,
        const size_t& lidarNo,
        const size_t& scanNo) const
{
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanNo;

    Description dp = lidar(scanPosNo, lidarNo);

    Description d;

    d.dataRoot = *dp.dataRoot + "/" + sstr.str();

    d.data = "points.ply";

    d.metaRoot = d.dataRoot;
    d.meta = "meta.yaml";

    return d;
}

} // namespace lvr2
