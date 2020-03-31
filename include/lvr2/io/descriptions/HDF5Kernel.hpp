#ifndef HDF5KERNEL_HPP
#define HDF5KERNEL_HPP

#include "lvr2/io/FileKernel.hpp"

namespace lvr2
{

class HDF5Kernel : public FileKernel<HDF5Kernel>
{
public:

    HDF5Kernel() = delete;
    HDF5Kernel(const std::string& hdf5file) : FileKernel(hdf5file) 
    {

    };

    virtual void saveMeshBuffer(
        const std::string& group, 
        const std::string& container, 
        const MeshBufferPtr& buffer)
    {

    }

    virtual void savePointBuffer(
        const std::string& group, 
        const std::string& container, 
        const PointBufferPtr& buffer)
    {

    }

    virtual void saveImage(
        const std::string& group, 
        const std::string& container,
        const cv::Mat& image)
    {

    }

    virtual void saveMetaYAML(
        const std::string& group, 
        const std::string& YAMLName,
        const YAML::Node& node)
    {

    }
   
    template<typename T>
    void saveArray(
        const std::string& group, 
        const std::string& container, 
        boost::shared_array<T> arr, 
        const size_t& length)
    {
        
    }

    void saveArray(
        const std::string& group, 
        const std::string& container, 
        boost::shared_array<T> arr, 
        const std::vector<size_t>& dims)
    {
        
    }

 
    virtual MeshBufferPtr loadMeshBuffer(
        const std::string& group, 
        const std::string container) = 0;

    virtual PointBufferPtr loadPointBuffer(
        const std::string& group,
        const std::string& container)
    {

    }

    virtual cv::Mat& loadImage(
        const std::string& group,
        const std::string& container)
    {

    }

    virtual YAML::Node& loadMetaYAML(
        const std::string& group,
        const std::string& container)
    {

    }

    template<typename T>
    boost::shared_array<float> loadArray(
        const std::string& group,
        const std::string& container,
        size_t& length)
    {
        return boost::shared_array(nullptr);
    }

    template<typename T>
    boost::shared_array<T> loadArray(
        const std::string& group,
        const std::string& container,
        std::vector<size_t>& dims)
    {
        return boost::shared_array(nullptr);
    }

}

#endif