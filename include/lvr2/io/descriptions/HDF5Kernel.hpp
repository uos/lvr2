#ifndef HDF5KERNEL_HPP
#define HDF5KERNEL_HPP

#include "lvr2/io/FileKernel.hpp"

namespace lvr2
{

class HDF5Kernel : public FileKernel
{
    HDF5Kernel() = delete;
    HDF5Kernel(const std::string& hdf5file) : FileKerner(hdf5file) {};


    virtual void saveMeshBuffer(const std::string& path, const MeshBufferPtr& buffer);
    virtual void savePointBuffer(const std::string& path, const PointBufferPtr& buffer);
    virtual void saveImage(const std::string& path, const cv::Mat& image);
    virtual void saveMetaNode(const std::string& path, const Node& node);
   
    virtual void saveFloatArray(
        const std::string& path, 
        const boost::shared_array<float>& arr, 
        const size_t& length);

    virtual void saveFloatArray(
        const std::string& path, 
        const boost::shared_array<float>& arr, 
        const std::vector<size_t>& dim);

    virtual void saveDoubleArray(
        const std::string& path, 
        const boost::shared_array<double>& arr, 
        const size_t& length);
        
    virtual void saveDoubleArray(
        const std::string& path, 
        const boost::shared_array<double>& arr, 
        const std::vector<size_t>& dim);

    virtual void saveUCharArray(
        const std::string& path, 
        const boost::shared_array<float>& arr, 
        const std::vector<size_t>& dim);

    virtual void saveUCharArray(
        const std::string& path, 
        const boost::shared_array<unsigned char>& arr, 
        const size_t& length);
        
    virtual void saveUCharArray(
        const std::string& path, 
        const boost::shared_array<unsigned char>& arr, 
        const std::vector<size_t>& dim);
 
    virtual MeshBufferPtr loadMeshBuffer(const std::string& path);
    virtual PointBufferPtr loadPointBuffer(const std::string& path);
    virtual cv::Mat& loadImage(const std::string& path);
    virtual Node& loadMetaNode(const std::string& path);

    virtual boost::shared_array<float> loadFloatArray(
        const std::string& path, size_t& length);
    virtual boost::shared_array<float> loadFloatArray(
        const std::string& path, std::vector<size_t>& dim); 

    virtual boost::shared_array<double> loadDoubleArray(
        const std::string& path, size_t& length);
    virtual boost::shared_array<double> loadDoubleArray(
        const std::string& path, std::vector<size_t>& dim); 

    virtual boost::shared_array<unsigned char> loadUCharArray(
        const std::string& path, size_t& length);
    virtual boost::shared_array<unsigned char> loadUCharArray(
        const std::string& path, std::vector<size_t>& dim); 
    
};

}

#endif