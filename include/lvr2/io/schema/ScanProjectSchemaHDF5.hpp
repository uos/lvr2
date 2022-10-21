#ifndef SCANPROJETSCHEMA_HDF5_HPP
#define SCANPROJETSCHEMA_HDF5_HPP

#include <string>

#include <boost/optional.hpp>
#include <boost/filesystem.hpp>

#include "lvr2/io/schema/ScanProjectSchema.hpp"

namespace lvr2
{

class ScanProjectSchemaHDF5 : public HDF5Schema
{
public:
    ScanProjectSchemaHDF5() : HDF5Schema() {};

    ~ScanProjectSchemaHDF5() = default;

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

#endif // SCANPROJETSCHEMA_HDF5_HPP_
