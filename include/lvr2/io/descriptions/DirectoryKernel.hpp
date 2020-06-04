#ifndef DIRECTORY_KERNEL_HPP
#define DIRECTORY_KERNEL_HPP

#include <boost/filesystem.hpp>
#include <iostream>
#include <regex>

#include "lvr2/io/descriptions/FileKernel.hpp"
#include "lvr2/io/descriptions/MetaFormatFactory.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/Timestamp.hpp"

namespace lvr2
{
    
class DirectoryKernel : public FileKernel
{
public:
    DirectoryKernel(const std::string &root) : FileKernel(root){};
    virtual ~DirectoryKernel() = default;

    virtual void saveMeshBuffer(
        const std::string& group, 
        const std::string& container, 
        const MeshBufferPtr& buffer) const;

    virtual void savePointBuffer(
        const std::string& group, 
        const std::string& container, 
        const PointBufferPtr& buffer) const;

    virtual void saveImage(
        const std::string& group, 
        const std::string& container,
        const cv::Mat& image) const;

    virtual void saveMetaYAML(
        const std::string& group, 
        const std::string& metaName,
        const YAML::Node& node) const;
   
    virtual MeshBufferPtr loadMeshBuffer(
        const std::string& group, 
        const std::string container) const;

    virtual PointBufferPtr loadPointBuffer(
        const std::string& group,
        const std::string& container) const;

    virtual boost::optional<cv::Mat> loadImage(
        const std::string& group,
        const std::string& container) const;

    virtual void loadMetaYAML(
        const std::string& group,
        const std::string& container,
        YAML::Node& node) const;

    virtual ucharArr loadUCharArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t> &dims) const;

    virtual floatArr loadFloatArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t> &dims) const;

    virtual doubleArr loadDoubleArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t> &dims) const;

    virtual void saveFloatArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<float>& data) const;

    virtual void saveDoubleArray(
        const std::string& groupName, const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<double>& data) const;

    virtual void saveUCharArray(
        const std::string& groupName, const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<unsigned char>& data) const;

    virtual bool exists(const std::string& group) const;
    virtual bool exists(const std::string& group, const std::string& container) const;

    virtual void subGroupNames(const std::string& group, std::vector<string>& subGroupNames) const;
    virtual void subGroupNames(const std::string& group, const std::regex& filter, std::vector<string>& subGroupNames) const;

protected:
    template <typename T>
    boost::shared_array<T> loadArray(const std::string &group, const std::string &constainer, std::vector<size_t> &dims) const
    {
        if (dims.size() > 0)
        {
            size_t length = dims[0];
            for (size_t i = 1; i < dims.size(); i++)
            {
                if (dims[i] != 0)
                {
                    length *= dims[i];
                }
                else
                {
                    std::cout << timestamp << "Warning: DirectoryKernel::LoadArray(): Found zero dim: " << i << std::endl;
                }
            }
            T *data = new T[length];
            std::ifstream in;
            for (size_t i = 0; i < length; i++)
            {
                in >> data[i];
            }
            return boost::shared_array<T>(data);
        }
        return boost::shared_array<T>(nullptr);
    }

    template <typename T>
    void saveArray(
        const std::string &group, const std::string &container, 
        const std::vector<size_t> &dims, const boost::shared_array<T>& data) const
    {
        if (dims.size() > 0)
        {
            size_t length = dims[0];
            for (size_t i = 1; i < dims.size(); i++)
            {
                if (dims[i] != 0)
                {
                    length *= dims[i];
                }
                else
                {
                    std::cout << timestamp << "Warning: DirectoryKernel::SaveArray(): Found zero dim: " << i << std::endl;
                }
            }
            
            std::ofstream out;
            for (size_t i = 0; i < length; i++)
            {
                out << data[i];
            }
        }
    }

    boost::filesystem::path getAbsolutePath(const std::string &group, const std::string &name) const;
};

using DirectoryKernelPtr = std::shared_ptr<DirectoryKernel>;

} // namespace lvr2

#endif