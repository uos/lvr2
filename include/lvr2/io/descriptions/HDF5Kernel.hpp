#ifndef HDF5KERNEL_HPP
#define HDF5KERNEL_HPP

#include <hdf5_hl.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

#include "lvr2/types/Channel.hpp"
#include "lvr2/io/descriptions/FileKernel.hpp"
#include "lvr2/io/hdf5/Hdf5Util.hpp"
#include "lvr2/types/VariantChannel.hpp"
#include "lvr2/io/descriptions/HDF5MetaDescriptionV2.hpp"

#include <type_traits>
#include <tuple>
namespace lvr2
{
class HDF5Kernel : public FileKernel
{
public:

    HDF5Kernel() = delete;
    HDF5Kernel(const std::string& hdf5file);
    ~HDF5Kernel() { delete m_metaDescription;}

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

    template<typename T>
    boost::shared_array<T> loadArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        size_t& size) const;

    template<typename T>
    boost::shared_array<T> loadArray(
        const std::string& groupName, 
        const std::string& datasetName, 
        std::vector<size_t>& dim) const;

    template<typename T> 
    void saveArray(
        const std::string& groupName, 
        const std::string& datasetName,
        const size_t& size,
        const boost::shared_array<T> data) const;

    template<typename T> 
    void saveArray(
        const std::string& groupName, 
        const std::string& datasetName,
        const vector<size_t>& dim,
        const boost::shared_array<T> data) const;

    template<typename T>
    ChannelOptional<T> loadChannelOptional(HighFive::Group& g, const std::string& datasetName) const;

    template<typename T>
    ChannelOptional<T> loadChannelOptional(const std::string& groupName, const std::string& datasetName) const;     

    // template<typename T>
    // ChannelOptional<T> load(
    //     HighFive::Group& g,
    //     std::string datasetName
    // ) const;

    template<typename T>
    void save(std::string groupName,
        std::string datasetName,
        const Channel<T>& channel) const;

    template<typename T>
    void save(HighFive::Group& g,
        std::string datasetName,
        const Channel<T>& channel) const;

    template<typename T>
    void save(HighFive::Group& g,
        std::string datasetName,
        const Channel<T>& channel,
        std::vector<hsize_t>& chunkSize) const;

     /**
     * @brief getChannel  Reads a float attribute channel in the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the float channel
     * @return            true if the channel has been loaded successfully, false otherwise
     */
    bool getChannel(const std::string group, const std::string name, FloatChannelOptional& channel) const;

    /**
     * @brief getChannel  Reads an index attribute channel in the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the index channel
     * @return            true if the channel has been loaded successfully, false otherwise
     */
    bool getChannel(const std::string group, const std::string name, IndexChannelOptional& channel) const;

    /**
     * @brief getChannel  Reads an unsigned char attribute channel in the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the unsigned char channel
     * @return            true if the channel has been loaded successfully, false otherwise
     */
    bool getChannel(const std::string group, const std::string name, UCharChannelOptional& channel) const;

    template <typename T>
    bool getChannel(const std::string group, const std::string name, boost::optional<AttributeChannel<T>>& channel) const;

    template <typename T>
    bool addChannel(const std::string group, const std::string name, const AttributeChannel<T>& channel) const;

    /**
     * @brief addChannel  Writes a float attribute channel from the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the float channel which should be written
     * @return            true if the channel has been written successfully, false otherwise
     */
    bool addChannel(const std::string group, const std::string name, const FloatChannel& channel) const;

    /**
     * @brief addChannel  Writes an index attribute channel from the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the index channel which should be written
     * @return            true if the channel has been written successfully, false otherwise
     */
    bool addChannel(const std::string group, const std::string name, const IndexChannel& channel) const;

    /**
     * @brief addChannel  Writes an unsigned char attribute channel from the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the unsigned char channel which should be written
     * @return            true if the channel has been written successfully, false otherwise
     */
    bool addChannel(const std::string group, const std::string name, const UCharChannel& channel) const;

    template<typename ...Tp>
    void save(std::string groupName, std::string datasetName, const VariantChannel<Tp...>& vchannel) const;
    
    template<typename ...Tp>
    void save(HighFive::Group& group, std::string datasetName, const VariantChannel<Tp...>& vchannel) const;
    
    template<typename VariantChannelT>
    boost::optional<VariantChannelT> load(std::string groupName, std::string datasetName) const;
    
    template<typename VariantChannelT>
    boost::optional<VariantChannelT> load(HighFive::Group& group, std::string datasetName) const;
    
    template<typename VariantChannelT>
    boost::optional<VariantChannelT> loadVariantChannel(std::string groupName, std::string datasetName) const;

    template<typename VariantChannelT>
    boost::optional<VariantChannelT> loadDynamic(HighFive::DataType dtype,
        HighFive::Group& group,
        std::string name) const;

    template<typename ...Tp>
    void saveDynamic(HighFive::Group& group,
        std::string datasetName,
        const VariantChannel<Tp...>& vchannel
    ) const;

    template<typename T>
    cv::Mat createMat(const std::vector<size_t>& dims) const;

    void loadMetaData(const YAML::Node& node);

    std::shared_ptr<HighFive::File>  m_hdf5File;

    HDF5MetaDescriptionBase* m_metaDescription;
   
};

using HDF5KernelPtr = std::shared_ptr<HDF5Kernel>;

} // namespace lvr2

#include "lvr2/io/descriptions/HDF5Kernel.tcc"

#endif