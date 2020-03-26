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

    virtual YAML::Node loadMetaYAML(
        const std::string& group,
        const std::string& container) const;

    virtual bool exists(const std::string& group) const;
    virtual bool exists(const std::string& group, const std::string& container) const;

    virtual void subGroupNames(const std::string& group, std::vector<string>& subGroupNames) const;
    virtual void subGroupNames(const std::string& group, const std::regex& filter, std::vector<string>& subGroupNames) const;

    template <typename T, DirectoryKernel>
    void saveArray(
        const std::string &group,
        const std::string &container,
        boost::shared_array<T> arr,
        const size_t &length) const
    {
        boost::filesystem::path p = getAbsolutePath(group, container);
        std::cout << timestamp << "Directory Kernel::saveArray(): " << p.string() << std::endl;
        std::ofstream out(p.c_str());
        for (size_t i = 0; i < length; i++)
        {
            out << arr[i];
        }
        out << std::endl;
        out.close();
    }

    template <typename T, DirectoryKernel>
    void saveArray(
        const std::string &group,
        const std::string &container,
        boost::shared_array<T> arr,
        const std::vector<size_t> &dims) const
    {
        boost::filesystem::path p = getAbsolutePath(group, container);
        std::cout << timestamp << "Directory Kernel::saveArray(): " << p.string() << std::endl;
        std::ofstream out(p.c_str());
        size_t length = dims[0] * dims[1];
        for (size_t i = 0; i < length; i++)
        {
            out << arr[i] << " ";
        }
        out << std::flush;
        out.close();
    }

    template <typename T, DirectoryKernel>
    boost::shared_array<float> loadArray(
        const std::string &group,
        const std::string &container,
        size_t &length) const
    {
        std::cout << timestamp << "DirectoryKernel::loadArray(length): Not implemented!" << std::endl;
        return boost::shared_array(nullptr);
    }

    template <typename T, DirectoryKernel>
    boost::shared_array<float> loadArray(
        const std::string &group,
        const std::string &container,
        std::vector<size_t> &dims) const
    {
        std::cout << timestamp << "DirectoryKernel::loadArray(dims): Not implemented!" << std::endl;
        return boost::shared_array(nullptr);
    }
protected:
    boost::filesystem::path getAbsolutePath(const std::string &group, const std::string &name) const;
};

} // namespace lvr2

#endif