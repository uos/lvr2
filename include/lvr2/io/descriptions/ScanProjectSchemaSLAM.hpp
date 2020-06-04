#ifndef ScanProjectSchemaSLAM_HPP
#define ScanProjectSchemaSLAM_HPP

#include "lvr2/io/descriptions/ScanProjectSchema.hpp"

namespace lvr2
{

class ScanProjectSchemaSLAM : public DirectorySchema
{
public:

    ScanProjectSchemaSLAM() {};

    virtual Description scanProject() const;
    virtual Description position(const size_t &scanPosNo) const;
    virtual Description scan(const size_t &scanPosNo, const size_t &scanNo) const;
    virtual Description scan(const std::string &scanPositionPath, const size_t &scanNo) const;

    virtual Description scanCamera(const size_t &scanPositionNo, const size_t &camNo) const;
    virtual Description scanCamera(const std::string &scanPositionPath, const size_t &camNo) const;

    virtual Description scanImage(
        const size_t &scanPosNo, const size_t &scanNo,
        const size_t &scanCameraNo, const size_t &scanImageNo) const;

    virtual Description scanImage(
        const std::string &scanImagePath, const size_t &scanImageNo) const;
};

}

#endif