#ifndef FILE_KERNEL_HPP
#define FILE_KERNEL_HPP

#include <string>
#include <vector>
#include <regex> 
#include <boost/optional.hpp>
#include <yaml-cpp/yaml.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/io/MeshBuffer.hpp"

namespace lvr2
{

class FileKernel
{
public:
    
    FileKernel() = delete;
    FileKernel(const std::string& res) : m_fileResourceName(res) {};

    virtual void saveMeshBuffer(
        const std::string& group, 
        const std::string& container, 
        const MeshBufferPtr& buffer) const = 0;

    virtual void savePointBuffer(
        const std::string& group, 
        const std::string& container, 
        const PointBufferPtr& buffer) const = 0;

    virtual void saveImage(
        const std::string& group, 
        const std::string& container,
        const cv::Mat& image) const = 0;

    virtual void saveMetaYAML(
        const std::string& group, 
        const std::string& metaName,
        const YAML::Node& node) const = 0;
   
    template<typename T, typename Implementation>
    void saveArray(
        const std::string& group, 
        const std::string& container, 
        boost::shared_array<T> arr, 
        const size_t& length) const
    {
        static_cast<Implementation*>(this)->saveArray(group, container, arr, length);
    }

    template<typename T, typename Implementation>
    void saveArray(
        const std::string& group, 
        const std::string& container, 
        boost::shared_array<T> arr, 
        const std::vector<size_t>& dims) const
    {
        static_cast<Implementation*>(this)->saveArray(group, container, arr, dims);
    }

 
    virtual MeshBufferPtr loadMeshBuffer(
        const std::string& group, 
        const std::string container) const = 0;

    virtual PointBufferPtr loadPointBuffer(
        const std::string& group,
        const std::string& container) const = 0;

    virtual boost::optional<cv::Mat> loadImage(
        const std::string& group,
        const std::string& container) const = 0;

    virtual YAML::Node loadMetaYAML(
        const std::string& group,
        const std::string& container) const = 0;

    template<typename T, typename Implementation>
    boost::shared_array<float>loadArray(
        const std::string& group,
        const std::string& container,
        size_t& length)  const
    {
        return static_cast<Implementation*>(this)->loadArray(group, container, length);
    }

    template<typename T, typename Implementation>
    boost::shared_array<float> loadArray(
        const std::string& group,
        const std::string& container,
        std::vector<size_t>& dims)
    {
        return static_cast<Implementation*>(this)->loadArray(group, container, dims);
    }

    virtual bool exists(const std::string& group) const = 0;
    virtual bool exists(const std::string& group, const std::string& container) const = 0;

    virtual void subGroupNames(const std::string& group, std::vector<string>& subGroupNames) const = 0;
    virtual void subGroupNames(const std::string& group, const std::regex& filter, std::vector<string>& subGroupNames) const = 0;

protected:
    std::string m_fileResourceName;
};

} // namespace lvr2

#endif