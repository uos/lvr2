#ifndef LABELSCANPROJECTSCHEMA_HDF5V2_HPP
#define LABELSCANPROJECTSCHEMA_HDF5V2_HPP

#include <string>

#include <boost/optional.hpp>
#include <boost/filesystem.hpp>

#include "lvr2/io/scanio/ScanProjectSchema.hpp"

namespace lvr2
{

class LabelScanProjectSchemaHDF5V2 : public LabelHDF5Schema
{
public:
    LabelScanProjectSchemaHDF5V2() = default;

    virtual ~LabelScanProjectSchemaHDF5V2() = default;
    virtual Description scanProject() const;
    virtual Description position(const size_t &scanPosNo) const;

    virtual Description scan(
        const size_t& scanPosNo,
        const size_t& lidarNo,
        const size_t& scanNo) const
    {
        return scan(scanPosNo, scanNo);
    }

    virtual Description scan(const size_t &scanPosNo, const size_t &scanNo) const;
    virtual Description scan(const std::string &scanPositionPath, const size_t &scanNo) const;
 
    virtual Description waveform(const size_t& scanPosNo, const size_t& scanNo) const;
    virtual Description waveform(const std::string& scanPositionPath, const size_t& scanNo) const;

    virtual Description camera(const size_t &scanPositionNo, const size_t &camNo) const;
    virtual Description camera(const std::string &scanPositionPath, const size_t &camNo) const;

    virtual Description cameraImage(
        const size_t &scanPosNo, 
        const size_t &scanCameraNo, const size_t &scanImageNo) const;

    virtual Description scanImage(
        const std::string &scanImagePath, const size_t &scanImageNo) const;

    virtual Description labelInstance(const std::string& group, const std::string& className, const std::string &instanceName) const;

    virtual Description lidar(const size_t& scanPosNo, const size_t& lidarNo) const
    {
        return Description();
    }

    virtual Description scanChannel(
        const size_t& scanPosNo,
        const size_t& lidarNo,
        const size_t& scanNo,
        const std::string& channelName) const
    {
        return Description();
    }

    virtual Description hyperspectralCamera(
        const size_t &scanPosNo,
        const size_t &camNo) const
    {
        return Description();
    }

    virtual Description hyperspectralPanorama(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& panoNo) const
    {
        return Description();
    }

    virtual Description hyperspectralPanoramaChannel(
        const size_t& scanPosNo,
        const size_t& camNo,
        const size_t& panoNo,
        const size_t& channelNo) const
    {
        return Description();
    }


};
} // namespace lvr2

#endif
