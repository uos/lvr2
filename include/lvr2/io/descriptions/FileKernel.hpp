#ifndef FILE_KERNEL_HPP
#define FILE_KERNEL_HPP

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "lvr2/types/MatrixTypes.hpp"

namespace lvr2
{

class FileKernel
{
public:
    
    FileKernel() = delete;
    FileKernel(const std::string& root) : m_root(root) {};

    virtual void saveMeshBuffer(const std::string& path, const MeshBufferPtr& buffer) = 0;
    virtual void savePointBuffer(const std::string& path, const PointBufferPtr& buffer) = 0;
    virtual void saveImage(const std::string& path, const cv::Mat& image) = 0;
    virtual void saveMetaNode(const std::string& path, const Node& node) = 0;
   
    virtual void saveFloatArray(
        const std::string& path, 
        const boost::shared_array<float>& arr, 
        const size_t& length) = 0;

    virtual void saveFloatArray(
        const std::string& path, 
        const boost::shared_array<float>& arr, 
        const std::vector<size_t>& dim) = 0;

    virtual void saveDoubleArray(
        const std::string& path, 
        const boost::shared_array<double>& arr, 
        const size_t& length) = 0;
        
    virtual void saveDoubleArray(
        const std::string& path, 
        const boost::shared_array<double>& arr, 
        const std::vector<size_t>& dim) = 0;

    virtual void saveUCharArray(
        const std::string& path, 
        const boost::shared_array<float>& arr, 
        const std::vector<size_t>& dim) = 0;

    virtual void saveUCharArray(
        const std::string& path, 
        const boost::shared_array<unsigned char>& arr, 
        const size_t& length) = 0;
        
    virtual void saveUCharArray(
        const std::string& path, 
        const boost::shared_array<unsigned char>& arr, 
        const std::vector<size_t>& dim) = 0;
 
    virtual MeshBufferPtr loadMeshBuffer(const std::string& path) = 0;
    virtual PointBufferPtr loadPointBuffer(const std::string& path) = 0;
    virtual cv::Mat& loadImage(const std::string& path) = 0;
    virtual Node& loadMetaNode(const std::string& path) = 0;

    virtual boost::shared_array<float> loadFloatArray(
        const std::string& path, size_t& length) = 0;
    virtual boost::shared_array<float> loadFloatArray(
        const std::string& path, std::vector<size_t>& dim) = 0; 

    virtual boost::shared_array<double> loadDoubleArray(
        const std::string& path, size_t& length) = 0;
    virtual boost::shared_array<double> loadDoubleArray(
        const std::string& path, std::vector<size_t>& dim) = 0; 

    virtual boost::shared_array<unsigned char> loadUCharArray(
        const std::string& path, size_t& length) = 0;
    virtual boost::shared_array<unsigned char> loadUCharArray(
        const std::string& path, std::vector<size_t>& dim) = 0; 
    
protected:
    std::string m_root;
};

} // namespace lvr2

#endif