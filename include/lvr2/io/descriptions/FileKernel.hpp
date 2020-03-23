#ifndef FILE_KERNEL_HPP
#define FILE_KERNEL_HPP

#include <string>
#include <vector>
#include <regex> 

#include <boost/optional.hpp>

#include <yaml-cpp/yaml.h>

#include "lvr2/types/MatrixTypes.hpp"

namespace lvr2
{

template<typename Implementation>
class FileKernel : public D
{
public:
    
    FileKernel() = delete;
    FileKernel(const std::string& res) : m_fileResourceName(res) {};

    virtual void saveMeshBuffer(
        const std::string& group, 
        const std::string& container, 
        const MeshBufferPtr& buffer) = 0;

    virtual void savePointBuffer(
        const std::string& group, 
        const std::string& container, 
        const PointBufferPtr& buffer) = 0;

    virtual void saveImage(
        const std::string& group, 
        const std::string& container,
        const cv::Mat& image) = 0;

    virtual void saveMetaYAML(
        const std::string& group, 
        const std::string& YAMLName,
        const YAML::Node& node) = 0;
   
    template<typename T>
    void saveArray(
        const std::string& group, 
        const std::string& container, 
        boost::shared_array<T> arr, 
        const size_t& length)
    {
        static_cast<Implementation*>(this)->saveArray(group, container, arr, length);
    }

    template<typename T>
    void saveArray(
        const std::string& group, 
        const std::string& container, 
        boost::shared_array<T> arr, 
        const std::vector<size_t>& dims)
    {
        static_cast<Implementation*>(this)->saveArray(group, container, arr, dims);
    }

 
    virtual MeshBufferPtr loadMeshBuffer(
        const std::string& group, 
        const std::string container) = 0;

    virtual PointBufferPtr loadPointBuffer(
        const std::string& group,
        const std::string& container) = 0;

    virtual boost::optional<cv::Mat> loadImage(
        const std::string& group,
        const std::string& container) = 0;

    virtual YAML::Node& loadMetaYAML(
        const std::string& group,
        const std::string& container) = 0;

    template<typename T>
    boost::shared_array<float> loadArray(
        const std::string& group,
        const std::string& container,
        size_t& length)
    {
        return static_cast<D*>(this)->loadArray(group, container, length);
    }

    template<typename T>
    boost::shared_array<float> loadArray(
        const std::string& group,
        const std::string& container,
        std::vector<size_t>& dims)
    {
        return static_cast<D*>(this)->loadArray(group, container, dims);
    }

    bool exists(const std::string& group) = 0;
    bool exists(const std::string& group, const std::string& container) = 0;

    void subGroupNames(const std::string& group, std::vector<string>& subGroupNames) = 0;
    void subGroupNames(const std::string& group, const std::regex& filter, std::vector<string>& subGroupNames) = 0;

protected:
    std::string m_fileResourceName;
};

} // namespace lvr2

#endif