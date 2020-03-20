#ifndef SCANPROJECTPARSER_HPP_
#define SCANPROJECTPARSER_HPP_

#include <string>
#include <tuple>

#include <boost/optional.hpp>

#include <yaml-cpp/yaml.h>
namespace lvr2
{

using StringOptional = boost::optional<std::string>;
using NodeOptional = boost::optional<YAML::Node>;
struct Description
{
    StringOptional groupName;
    StringOptional dataSetName;
    NodeOptional metaData;
};

class ScanProjectStructure
{
public:
    ScanProjectStructure() = delete;

    ScanProjectStructure(const std::string& root) 
        : m_root(root), m_lastScanPosition(0), m_lastScan(0) {};

    ~ScanProjectStructure() = default;

    virtual Description scanProject() = 0;
    virtual Description position(const size_t& scanPosNo) = 0;
    virtual Description scan(const size_t& scanPosNo, const size_t& scanNo) = 0;
    virtual Description scan(const std::string& scanPositionPath, const size_t& scanNo) = 0;
    
    virtual Description scanCamera(const size_t& scanPositionNo, const size_t& camNo) = 0;
    virtual Description scanCamera(const std::string& scanPositionPath, const size_t& camNo) = 0;
 
    virtual Description scanImage(
        const size_t& scanPosNo, const size_t& scanNo,
        const size_t& scanCameraNo, const size_t& scanImageNo) = 0;

    virtual Description scanImage(
        const std::string& scanImagePath, const size_t& scanImageNo) = 0;

protected:
    std::string     m_root;
    size_t          m_lastScanPosition;
    size_t          m_lastScan;
};

}

#endif