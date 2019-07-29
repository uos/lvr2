#ifndef __DIRECTORY_PARSER_HPP__
#define __DIRECTORY_PARSER_HPP__

#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <Eigen/Dense>

#include "Timestamp.hpp"

namespace lvr2
{

struct ScanInfo
{
    string              m_filename;
    size_t              m_numPoints;
    Eigen::Matrix4d     m_pose;
};

class ScanDirectoryParser
{
   
public:
    ScanDirectoryParser(const std::string& directory) noexcept;

    void setPointCloudPrefix(const std::string& prefix);
    void setPointCloudExtension(const std::string& extension);
    void setPosePrefix(const std::string& prefix);
    void setPoseExtension(const std::string& extension);

    void setStart(int s);
    void setEnd(int e);

    void setTargetSize(const size_t& size);

    void parseDirectory();

    ~ScanDirectoryParser() = default;

private:

    using Path = boost::filesystem::path;

    size_t examinePLY(const std::string& filename);
    size_t examineASCII(const std::string& filename);    

    Eigen::Matrix4d getPose(const Path& poseFile);

    size_t                  m_numPoints;
    std::string             m_pointPrefix;
    std::string             m_posePrefix;
    std::string             m_poseExtension;
    std::string             m_pointExtension;
    std::string             m_directory;

    size_t                  m_start;
    size_t                  m_end;
    size_t                  m_targetSize;

    std::vector<ScanInfo>   m_scans;
};

} // namespace lvr2

#endif