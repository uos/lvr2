#ifndef SCANPROJETSCHEMA_HDF5_HPP
#define SCANPROJETSCHEMA_HDF5_HPP

#include <string>

#include <boost/optional.hpp>
#include <boost/filesystem.hpp>

#include "lvr2/io/descriptions/ScanProjectSchema.hpp"

namespace lvr2
{

class ScanProjectSchemaHDF5 : public HDF5Schema
{
public:
    ScanProjectSchemaHDF5() : HDF5Schema() {};

    ~ScanProjectSchemaHDF5() = default;

    virtual StringOptional scanProjectData() const;

    virtual StringOptional positionData(
        const size_t& scanPosNo) const;

    virtual StringOptional lidarData(
        const size_t& scanPosNo,
        const size_t& lidarNo) const;
    
    virtual StringOptional scanData(
        const size_t& scanPosNo,
        const size_t& lidarNo,
        const size_t& scanNo) const;

    virtual StringOptional scanChannelData(
        const size_t& scanPosNo,
        const size_t& lidarNo,
        const size_t& scanNo,
        const std::string& channelName) const;

    virtual StringOptional cameraData(
        const size_t& scanPosNo,
        const size_t& camNo) const;
 
    virtual StringOptional cameraImageData(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& cameraImageNo) const;

    virtual StringOptional hyperspectralCameraData(
        const size_t& scanPosNo,
        const size_t& camNo) const;

    virtual StringOptional hyperspectralPanoramaData(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& panoNo) const;

    virtual StringOptional hyperspectralPanoramaChannelData(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& panoNo,
        const size_t& channelNo) const;
};

} // namespace lvr2

#endif // SCANPROJETSCHEMA_HDF5_HPP_
