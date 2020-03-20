#ifndef SCANPROJECTPARSER_HYPERLIB_HPP_
#define SCANPROJECTPARSER_HYPERLIB_HPP_

#include <string>

#include <boost/optional.hpp>
#include <boost/filesystem.hpp>

#include "lvr2/io/descriptions/ScanProjectStructure.hpp"

namespace lvr2
{

class ScanProjectStructureHyperlib : public ScanProjectStructure
{
public:
    ScanProjectStructureHyperlib() = delete;

    ScanProjectStructureHyperlib(const std::string& root) 
        : ScanProjectStructure(m_root),
          m_rootPath(root), m_metaPath("meta.yaml") 
        {
        
        };

    ~ScanProjectStructureHyperlib() = default;

    virtual Description scanProject() = 0;
    virtual Description position(const size_t &scanPosNo) = 0;
    virtual Description scan(const size_t &scanPosNo, const size_t &scanNo) = 0;
    virtual Description scan(const std::string &scanPositionPath, const size_t &scanNo) = 0;

    virtual Description scanCamera(const size_t &scanPositionNo, const size_t &camNo) = 0;
    virtual Description scanCamera(const std::string &scanPositionPath, const size_t &camNo) = 0;

    virtual Description scanImage(
        const size_t &scanPosNo, const size_t &scanNo,
        const size_t &scanCameraNo, const size_t &scanImageNo) = 0;

    virtual Description scanImage(
        const std::string &scanImagePath, const size_t &scanImageNo) = 0;

private:
    boost::filesystem::path m_rootPath;
    boost::filesystem::path m_metaPath;
};

} // namespace lvr2

#endif