#ifndef DIRECTORY_KERNEL_HPP
#define DIRECTORY_KERNEL_HPP

#include <boost/filesystem.hpp>
#include <iostream>
#include <regex>
#include <iostream>

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

    virtual intArr loadIntArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t> &dims) const;
    
    virtual uint16Arr loadUInt16Array(
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
    
    virtual void saveIntArray(
        const std::string& groupName, const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<int>& data) const;

    virtual void saveUInt16Array(
        const std::string& groupName, const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<uint16_t>& data) const;

    void saveFloatChannel(
        const std::string& group, const std::string& name, 
        const Channel<float>& channel);

    virtual bool exists(const std::string& group) const;
    virtual bool exists(const std::string& group, const std::string& container) const;

    virtual void subGroupNames(const std::string& group, std::vector<string>& subGroupNames) const;
    virtual void subGroupNames(const std::string& group, const std::regex& filter, std::vector<string>& subGroupNames) const;

    
protected:
    template <typename T>
    boost::shared_array<T> loadArray(const std::string &group, const std::string &container, std::vector<size_t> &dims) const
    {
        boost::filesystem::path rootPath(m_fileResourceName);
        boost::filesystem::path groupPath(group);
        boost::filesystem::path containerPath(container);
        boost::filesystem::path finalPath(rootPath / groupPath / containerPath);   

        std::ifstream in(finalPath.string(), std::ios::in | std::ios::binary);
        if (!in.good())
        {
            return boost::shared_array<T>(nullptr);
        }

        //read Dimensions size
        size_t dimSize;
        size_t totalArraySize = 0;
        in.read(reinterpret_cast<char *>(&dimSize), sizeof(dimSize));


        for(int i = 0; i < dimSize; i++)
        {
            size_t tmp;
            in.read(reinterpret_cast<char *>(&tmp), sizeof(tmp));
            if(totalArraySize == 0)
            {
                totalArraySize = tmp;
            } else{
                totalArraySize *= tmp;
            }
            dims.push_back(tmp);
        }
                std::cout << "DimSize " << dims[0] << "  " <<dims[1] << std::endl;
         T* rawData = new T[totalArraySize];

        in.read(reinterpret_cast<char *>(rawData), totalArraySize * sizeof(T));
        boost::shared_array<T> ret(rawData);   
        return ret;
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
            
            boost::filesystem::path p = getAbsolutePath(group, container);
            if(!boost::filesystem::exists(p.parent_path()))
            {
                boost::filesystem::create_directories(p.parent_path());
            }

            std::cout << "Writing Array of size " << length << " to " << p << std::endl;
            
            // 1. HEADER
            // size_t type_id = PointBuffer::index_of_type<T>::value;
            // std::ofstream fout_header(p.string() + ".def");
            // fout_header << type_id;

            // for(size_t i=0; i<dims.size(); i++)
            // {
            //     fout_header << dims[i];
            // }

            // fout_header.close();

            // 2. DATA
            std::ofstream fout_data(p.string() + ".data", std::ios::binary);
            for (size_t i = 0; i < length; i++)
            {
                fout_data << data[i];
            }

            fout_data.close();
        }
    }

    boost::filesystem::path getAbsolutePath(const std::string &group, const std::string &name) const;
};

using DirectoryKernelPtr = std::shared_ptr<DirectoryKernel>;

} // namespace lvr2

#endif