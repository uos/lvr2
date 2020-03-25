#ifndef DIRECTORY_KERNEL_HPP
#define DIRECTORY_KERNEL_HPP

#include <boost/filesystem.hpp>
#include "lvr2/io/descriptions/FileKernel.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/Timestamp.hpp"

namespace lvr2
{
    
class DirectoryKernel : public FileKernel<DirectoryKernel>
{
public:

    DirectoryKernel(const std::string& root) : FileKernel<DirectoryKernel>(root) {};

    virtual void saveMeshBuffer(
        const std::string& group, 
        const std::string& container, 
        const MeshBufferPtr& buffer) const
    {
        boost::fileystem::path p = getAbsolutePath(group, container);
        ModelPtr model(new Model);
        model->m_mesh = buffer;
        std::cout << timestamp << "Directory Kernel::saveMeshBuffer(): " << p.string() << std::endl();
        ModelFactory::saveModel(model, p.string());
    }

    virtual void savePointBuffer(
        const std::string& group, 
        const std::string& container, 
        const PointBufferPtr& buffer) const
    {
        boost::fileystem::path p = getAbsolutePath(group, container);
        ModelPtr model(new Model);
        model->m_pointCloud = buffer;
        std::cout << timestamp << "Directory Kernel::savePointBuffer(): " << p.string() << std::endl();
        ModelFactory::saveModel(model, p.string());
    }

    virtual void saveImage(
        const std::string& group, 
        const std::string& container,
        const cv::Mat& image) const
    {
        boost::fileystem::path p = getAbsolutePath(group, container);
        std::cout << timestamp << "Directory Kernel::saveImage(): " << p.string() << std::endl();

        cv::imwrite(p.string(), image);
    }

    virtual void saveMetaYAML(
        const std::string& group, 
        const std::string& container,
        const YAML::Node& node) const
    {
        p.
    }
   
    template<typename T>
    void saveArray(
        const std::string& group, 
        const std::string& container, 
        boost::shared_array<T> arr, 
        const size_t& length) const
    {
        
    }

    void saveArray(
        const std::string& group, 
        const std::string& container, 
        boost::shared_array<T> arr, 
        const std::vector<size_t>& dims) const
    {
        
    }

 
    virtual MeshBufferPtr loadMeshBuffer(
        const std::string& group, 
        const std::string container) = 0;

    virtual PointBufferPtr loadPointBuffer(
        const std::string& group,
        const std::string& container) const
    {

    }

    virtual cv::Mat& loadImage(
        const std::string& group,
        const std::string& container) const
    {

    }

    virtual YAML::Node& loadMetaYAML(
        const std::string& group,
        const std::string& container) const
    {

    }

    template<typename T>
    boost::shared_array<float> loadArray(
        const std::string& group,
        const std::string& container,
        size_t& length) const
    {
        return boost::shared_array(nullptr);
    }

    template<typename T>
    boost::shared_array<float> loadArray(
        const std::string& group,
        const std::string& container,
        std::vector<size_t>& dims) const
    {
        return boost::shared_array(nullptr);
    }

    bool exists(const std::string& group) const
    {
        return true;
    }
    bool exists(const std::string& group, const std::string& container) const
    {
        return true;
    }

    void subGroupNames(const std::string& group, std::vector<string>& subGroupNames) const
    {

    }

    void subGroupNames(const std::string& group, const std::regex& filter, std::vector<string>& subGroupNames) const
    {

    }

private:
    boost::filesystem::path getAbsolutePath(const std::string& group, const std::string& name) const
    {
        boost::filesystem::path groupPath(group);
        boost::filesystem::path namePath(name);
        boost::filesystem::path rootPath(m_fileResourceName);
        boost::filesystem::path ret = rootPath / groupPath / namePath;
        return ret;
    }

};

} // namespace lvr2

#endif