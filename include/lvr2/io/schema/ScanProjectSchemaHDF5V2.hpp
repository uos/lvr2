#ifndef SCANPROJECTSCHEMA_HDF5V2_HPP
#define SCANPROJECTSCHEMA_HDF5V2_HPP

#include <string>

#include <boost/optional.hpp>
#include <boost/filesystem.hpp>

#include "lvr2/io/schema/ScanProjectSchema.hpp"

namespace lvr2
{

class ScanProjectSchemaHDF5V2 : public HDF5Schema
{
public:
    ScanProjectSchemaHDF5V2() {};

    ~ScanProjectSchemaHDF5V2() = default;

    virtual Description scanProject() const;
    virtual Description position(const size_t &scanPosNo) const;
    virtual Description scan(const size_t &scanPosNo, const size_t &scanNo) const;
    virtual Description scan(const std::string &scanPositionPath, const size_t &scanNo) const;
 
    virtual Description waveform(const size_t& scanPosNo, const size_t& scanNo) const;
    virtual Description waveform(const std::string& scanPositionPath, const size_t& scanNo) const;

    virtual Description scanCamera(const size_t &scanPositionNo, const size_t &camNo) const;
    virtual Description scanCamera(const std::string &scanPositionPath, const size_t &camNo) const;

    virtual Description scanImage(
        const size_t &scanPosNo, 
        const size_t &scanCameraNo, const size_t &scanImageNo) const;

    virtual Description scanImage(
        const std::string &scanImagePath, const size_t &scanImageNo) const;

    virtual Description cameraImage(
        const size_t &scanPosNo,
        const size_t &groupNo,
        const size_t &camNo,
        const size_t &imgNo) const override;

    virtual Description cameraImageGroup(
        const size_t &scanPosNo,
        const size_t &camNo,
        const size_t &GroupNo) const override;
};

} // namespace lvr2

#endif
