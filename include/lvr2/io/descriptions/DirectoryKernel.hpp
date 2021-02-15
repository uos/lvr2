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

    virtual charArr loadCharArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t> &dims) const;

    virtual ucharArr loadUCharArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t> &dims) const;

    virtual shortArr loadShortArray(
        const std::string& group, 
        const std::string& constainer, 
        std::vector<size_t>& dims) const;

    virtual ushortArr loadUShortArray(
        const std::string& group, 
        const std::string& constainer, 
        std::vector<size_t>& dims) const;

    virtual uint16Arr loadUInt16Array(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t> &dims) const;

    virtual intArr loadIntArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t> &dims) const;

    virtual uintArr loadUIntArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t> &dims) const;

    virtual lintArr loadLIntArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t> &dims) const;

    virtual ulintArr loadULIntArray(
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

    virtual boolArr loadBoolArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t> &dims) const;

    virtual void saveCharArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<char>& data) const;

    virtual void saveUCharArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<unsigned char>& data) const;

    virtual void saveShortArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<short>& data) const;

    virtual void saveUShortArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<unsigned short>& data) const;

    virtual void saveUInt16Array(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<uint16_t>& data) const;

    virtual void saveIntArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<int>& data) const;

    virtual void saveUIntArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<unsigned int>& data) const;

    virtual void saveLIntArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<long int>& data) const;

    virtual void saveULIntArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<unsigned long int>& data) const;

    virtual void saveFloatArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<float>& data) const;

    virtual void saveDoubleArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<double>& data) const;

    virtual void saveBoolArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        const std::vector<size_t>& dimensions, 
        const boost::shared_array<bool>& data) const;

    virtual bool exists(const std::string& group) const;
    virtual bool exists(const std::string& group, const std::string& container) const;

    virtual void subGroupNames(const std::string& group, std::vector<string>& subGroupNames) const;
    virtual void subGroupNames(const std::string& group, const std::regex& filter, std::vector<string>& subGroupNames) const;

    // TODO: metas with "sensor_type" filter
    virtual std::unordered_map<std::string, YAML::Node> metas(
        const std::string& group) const;

    virtual std::unordered_map<std::string, YAML::Node> metas(
        const std::string& group, const std::string& sensor_type
    ) const;

    virtual bool isMeta(const std::string& path) const;

protected:
    template <typename T>
    boost::shared_array<T> loadArray(
        const std::string &group, 
        const std::string &container, 
        std::vector<size_t> &dims) const
    {
        dims.resize(0);

        boost::filesystem::path p = getAbsolutePath(group, container);

        if(p.extension() == "")
        {
            p += ".data";
        }

        if(!boost::filesystem::exists(p))
        {
            // return empty pointer if path not exist:
            // should not happen. the check needs to be done from above module
            return boost::shared_array<T>(nullptr);;
        }

        std::string filename = p.string();

        if(p.extension() == ".data")
        {

            std::ifstream in(filename, std::ios::in | std::ios::binary);
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
            
            T* rawData = new T[totalArraySize];

            in.read(reinterpret_cast<char *>(rawData), totalArraySize * sizeof(T));
            boost::shared_array<T> ret(rawData);
            return ret;

        } else {

            // has some unknown extension
            PointBufferPtr points = loadPointBuffer(group, container);

            std::cout << "Loaded " << points->numPoints() << " Points" << std::endl;

            return boost::shared_array<T>(nullptr);
        }
    }

    template <typename T>
    void saveArray(
        const std::string &group, const std::string &container, 
        const std::vector<size_t> &dims, const boost::shared_array<T>& data) const
    {
        std::cout << "SAVE ARRAY TO " << group << "; " << container << std::endl;

        

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

            std::string filename = p.string();
            if(p.extension() != "")
            {
                std::cout << "Found extrension: " << p.extension() << std::endl;
                // find out how to handle it
                PointBufferPtr buffer(new PointBuffer);
                (*buffer)["points"] = Channel<T>(dims[0], dims[1], data);
                savePointBuffer(group, container, buffer);
            } else {
                filename += ".data";
                std::ofstream fout_data(filename, std::ios::out | std::ios::binary);
            
                // dims
                size_t ndims = dims.size();
                fout_data.write(reinterpret_cast<const char *>(&ndims), sizeof(size_t));

                for(size_t i=0; i<ndims; i++)
                {
                    fout_data.write(reinterpret_cast<const char *>(&dims[i]), sizeof(size_t));
                }

                // data
                fout_data.write(reinterpret_cast<const char *>(&data[0]), sizeof(T) * length);


                fout_data.close();
            }

            
        }
    }

    boost::filesystem::path getAbsolutePath(const std::string &group, const std::string &name) const;
};

using DirectoryKernelPtr = std::shared_ptr<DirectoryKernel>;

} // namespace lvr2

#endif