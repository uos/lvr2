#ifndef FILE_KERNEL_HPP
#define FILE_KERNEL_HPP

#include <string>
#include <vector>
#include <regex> 
#include <boost/optional.hpp>
#include <yaml-cpp/yaml.h>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/util/Tuple.hpp"

namespace lvr2
{

class FileKernel
{
public:
    using ImplementedTypes = Tuple<
        char,
        unsigned char,
        short,
        unsigned short,
        uint16_t,
        int,
        unsigned int,
        long int,
        unsigned long int, // size_t
        float,
        double,
        bool
    >;
    
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

    virtual charArr loadCharArray(
        const std::string& group,
        const std::string& container,
        std::vector<size_t>& dims) const = 0;

    virtual ucharArr loadUCharArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t>& dims) const = 0;

    virtual shortArr loadShortArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t>& dims) const = 0;

    virtual ushortArr loadUShortArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t>& dims) const = 0;

    virtual uint16Arr loadUInt16Array(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t>& dims) const = 0;
    
    virtual intArr loadIntArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t>& dims) const = 0;

    virtual uintArr loadUIntArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t>& dims) const = 0;

    virtual lintArr loadLIntArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t>& dims) const = 0;

    virtual ulintArr loadULIntArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t>& dims) const = 0;

    virtual floatArr loadFloatArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t>& dims) const = 0;

    virtual doubleArr loadDoubleArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t>& dims) const = 0;

    virtual boolArr loadBoolArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t> &dims) const = 0;

    // Shortcut
    template<typename T>
    boost::shared_array<T> loadArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t>& dims) const;

    // Saving interface
    virtual void saveCharArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<char>& data) const = 0;

    virtual void saveUCharArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<unsigned char>& data) const = 0;

    virtual void saveShortArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<short>& data) const = 0;

    virtual void saveUShortArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<unsigned short>& data) const = 0;

    virtual void saveUInt16Array(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<uint16_t>& data) const = 0;

    virtual void saveIntArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<int>& data) const = 0;

    virtual void saveUIntArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<unsigned int>& data) const = 0;

    virtual void saveLIntArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<long int>& data) const = 0;

    virtual void saveULIntArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<unsigned long int>& data) const = 0;

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

    virtual void saveBoolArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<bool>& data) const = 0;

    // shortcut
    template<typename T>
    void saveArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<T>& data) const;

    virtual bool exists(const std::string& group) const = 0;
    virtual bool exists(const std::string& group, const std::string& container) const = 0;

    virtual void subGroupNames(const std::string& group, std::vector<string>& subGroupNames) const = 0;
    virtual void subGroupNames(const std::string& group, const std::regex& filter, std::vector<string>& subGroupNames) const = 0;

    virtual std::vector<std::string> listDatasets(const std::string& group) const = 0;

    // TODO: make pure virtual
    virtual std::unordered_map<std::string, YAML::Node> metas(
        const std::string& group) const
    {
        std::unordered_map<std::string, YAML::Node> ret;
        return ret;
    }

    // TODO: make pure virtual
    virtual std::unordered_map<std::string, YAML::Node> metas(
        const std::string& group, const std::string& sensor_type) const
    {
        std::unordered_map<std::string, YAML::Node> ret;
        return ret;
    }

    // TODO: make pure virtual
    virtual bool isMeta(const std::string& path) const
    {
        return false;
    }
    /**
     * @brief   Returns the path to the file resource of the 
     *          kernel
     */
    std::string fileResource() const { return m_fileResourceName; }

protected:
    std::string m_fileResourceName;
};

using FileKernelPtr = std::shared_ptr<FileKernel>;

} // namespace lvr2

#include "FileKernel.tcc"

#endif // FileKernel