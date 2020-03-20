#ifndef SCANPROJECTSTRUCTURESLAM_HPP
#define SCANPROJECTSTRUCTURESLAM_HPP

#include "lvr2/io/descriptions/ScanProjectStructure.hpp"

namespace lvr2
{

class ScanProjectStructureSLAM : public ScanProjectStructure
{
public:
    ScanProjectStructureSLAM() = delete;
    ScanProjectStructureSLAM(const std::string& root) : ScanProjectStructure(root) {}

    virtual Description scanProject();
    virtual Description position(const size_t &scanPosNo);
    virtual Description scan(const size_t &scanPosNo, const size_t &scanNo);
    virtual Description scan(const std::string &scanPositionPath, const size_t &scanNo);

    virtual Description scanCamera(const size_t &scanPositionNo, const size_t &camNo);
    virtual Description scanCamera(const std::string &scanPositionPath, const size_t &camNo);

    virtual Description scanImage(
        const size_t &scanPosNo, const size_t &scanNo,
        const size_t &scanCameraNo, const size_t &scanImageNo);

    virtual Description scanImage(
        const std::string &scanImagePath, const size_t &scanImageNo);
};

}

#endif