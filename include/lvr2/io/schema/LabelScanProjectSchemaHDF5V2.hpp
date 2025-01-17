#ifndef LABELSCANPROJECTSCHEMAHDF5V2
#define LABELSCANPROJECTSCHEMAHDF5V2

#include <string>

#include <boost/optional.hpp>
#include <boost/filesystem.hpp>

#include "lvr2/io/schema/ScanProjectSchema.hpp"

namespace lvr2
{

class LabelScanProjectSchemaHDF5V2 : public LabelHDF5Schema
{
public:
    LabelScanProjectSchemaHDF5V2() = default;

    virtual ~LabelScanProjectSchemaHDF5V2() = default;
    virtual Description scanProject() const
    {
        return Description();
    }

    virtual Description position(const size_t &scanPosNo) const
    {
        return Description();
    }

    virtual Description scan(
        const size_t& scanPosNo,
        const size_t& lidarNo,
        const size_t& scanNo) const
    {
        return scan(scanPosNo, scanNo);
    }

    virtual Description scan(const size_t &scanPosNo, const size_t &scanNo) const
    {
        return Description();
    }

    virtual Description scan(const std::string &scanPositionPath, const size_t &scanNo) const
    {
        return Description();
    }
 
    virtual Description waveform(const size_t& scanPosNo, const size_t& scanNo) const
    {
        return Description();
    }

    virtual Description waveform(const std::string& scanPositionPath, const size_t& scanNo) const
    {
        return Description();
    }

    virtual Description camera(const size_t &scanPositionNo, const size_t &camNo) const
    {
        return Description();
    }

    virtual Description camera(const std::string &scanPositionPath, const size_t &camNo) const
    {
        return Description();
    }

    virtual Description cameraImage(
        const size_t &scanPosNo,
        const size_t &camNo,
        const size_t &GroupNo,
        const size_t &imgNo) const override
    {
        return Description();
    }

    virtual Description cameraImageGroup(
        const size_t &scanPosNo,
        const size_t &camNo,
        const size_t &GroupNo) const override
    {
        return Description();
    }

    virtual Description scanImage(
        const std::string &scanImagePath, const size_t &scanImageNo) const
    {
        return Description();
    }

    virtual Description labelInstance(const std::string& group, const std::string& className, const std::string &instanceName) const
    {
        return Description();
    }

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

    virtual Description hyperspectralPanoramaPreview(
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

#endif // LABELSCANPROJECTSCHEMAHDF5V2
