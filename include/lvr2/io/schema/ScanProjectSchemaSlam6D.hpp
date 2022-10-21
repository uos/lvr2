#ifndef SCANPROJECTSCHEMASLAM6D
#define SCANPROJECTSCHEMASLAM6D

#include <string>

#include <boost/optional.hpp>
#include <boost/filesystem.hpp>

#include "lvr2/io/schema/ScanProjectSchema.hpp"

namespace lvr2
{

/**
 * @brief ScanProjectSchema for Slam6D projects
 * 
 * Takes the transformation and poseEstimation from scanPositions and 
 * uses the data from first scanner (0) and first scan(0) of the scanPositions
 * 
 * That assumes that the scan is recorded at the origin of the scanPosition
 * Otherwise, you have to adjust the scanPositions transformation such that
 * it is given
 * 
 */
class ScanProjectSchemaSlam6D : public DirectorySchema
{
public:
    ScanProjectSchemaSlam6D(const std::string& rootDir) : DirectorySchema(rootDir) {};

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
        const size_t& groupNo,
        const size_t& camNo,
        const size_t& imgNo) const override;

    virtual Description cameraImageGroup(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& GroupNo) const override;


    virtual Description hyperspectralCamera(
        const size_t& scanPosNo,
        const size_t& camNo) const;

    virtual Description hyperspectralPanorama(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& panoNo) const;

    virtual Description hyperspectralPanoramaPreview(
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

#endif // SCANPROJECTSCHEMASLAM6D
