#ifndef SCANPROJECTSCHEMA_SLAM6D_HPP_
#define SCANPROJECTSCHEMA_SLAM6D_HPP_

#include <string>

#include <boost/optional.hpp>
#include <boost/filesystem.hpp>

#include "lvr2/io/descriptions/ScanProjectSchema.hpp"

namespace lvr2
{

class ScanProjectSchemaSlam6D : public DirectorySchema
{
public:
    ScanProjectSchemaSlam6D(std::string& rootDir) : DirectorySchema(rootDir) {};

    ~ScanProjectSchemaSlam6D() = default;

    virtual Description scanProject() const;

    virtual Description position(
        const size_t& scanPosNo) const;

    virtual Description lidar(
        const size_t& scanPosNo,
        const size_t& lidarNo) const;
    
    virtual Description scan(
        const size_t& scanPosNo,
        const size_t& lidarNo,
        const size_t& scanNo) const;

    virtual Description scanChannel(
        const size_t& scanPosNo,
        const size_t& lidarNo,
        const size_t& scanNo,
        const std::string& channelName) const;

    virtual Description camera(
        const size_t& scanPosNo,
        const size_t& camNo) const;
 
    virtual Description cameraImage(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& cameraImageNo) const;

    virtual Description hyperspectralCamera(
        const size_t& scanPosNo,
        const size_t& camNo) const;

    virtual Description hyperspectralPanorama(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& panoNo) const;

    virtual Description hyperspectralPanoramaChannel(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& panoNo,
        const size_t& channelNo) const;
};

} // namespace lvr2

#endif // SCANPROJECTSCHEMA_SLAM6D_HPP_