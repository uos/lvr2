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

    virtual MeshBufferPtr loadMeshBuffer(
        const std::string& group, 
        const std::string container) const = 0;

    virtual PointBufferPtr loadPointBuffer(
        const std::string& group,
        const std::string& container) const = 0;

    virtual boost::optional<cv::Mat> loadImage(
        const std::string& group,
        const std::string& container) const = 0;

    /// That we don't return the YAML node is on purpose
    /// to use the initial structure to look for the
    /// fields that should be loaded!
    virtual void loadMetaYAML(
        const std::string& group,
        const std::string& container,
        YAML::Node& node) const = 0;

    virtual ucharArr loadUCharArray(
        const std::string& group, 
        const std::string& constainer, 
        std::vector<size_t>& dims) const = 0;

    virtual floatArr loadFloatArray(
        const std::string& group, 
        const std::string& constainer, 
        std::vector<size_t>& dims) const = 0;

    virtual doubleArr loadDoubleArray(
        const std::string& group, 
        const std::string& constainer, 
        std::vector<size_t>& dims) const = 0;

    virtual void saveFloatArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<float>& data) const = 0;

    virtual void saveDoubleArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<double>& data) const = 0;

    virtual void saveUCharArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<unsigned char>& data) const = 0;

    virtual bool exists(const std::string& group) const = 0;
    virtual bool exists(const std::string& group, const std::string& container) const = 0;

    virtual void subGroupNames(const std::string& group, std::vector<string>& subGroupNames) const = 0;
    virtual void subGroupNames(const std::string& group, const std::regex& filter, std::vector<string>& subGroupNames) const = 0;

protected:
    std::string m_fileResourceName;
};

using FileKernelPtr = std::shared_ptr<FileKernel>;

} // namespace lvr2

#endif