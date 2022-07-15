#ifndef HDF5KERNEL
#define HDF5KERNEL

#include "lvr2/types/VariantChannel.hpp"
#include "lvr2/types/Channel.hpp"
#include "lvr2/util/Hdf5Util.hpp"

#include "lvr2/io/kernels/FileKernel.hpp"

namespace lvr2
{


/**
 * @brief HDF5Kernel configurations.
 * 
 * Consists of parameters that influence the general way of storing
 * 
 */
struct HDF5KernelConfig {
    /// The higher the compressionLevel the lower the memory consumption but higher runtime
    unsigned int compressionLevel = 9;
};

class HDF5Kernel : public FileKernel
{
public:

    HDF5Kernel() = delete;
    HDF5Kernel(const std::string& hdf5file, HDF5KernelConfig config = HDF5KernelConfig());
    ~HDF5Kernel() {
         if(m_hdf5File)
         {
             m_hdf5File->flush();
         }
    }

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

    virtual bool loadMetaYAML(
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
        const std::string& constainer, 
        std::vector<size_t>& dims) const;

    virtual lintArr loadLIntArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t>& dims) const;

    virtual ulintArr loadULIntArray(
        const std::string& group, 
        const std::string& container, 
        std::vector<size_t>& dims) const;

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

    virtual std::vector<std::string> listDatasets(const std::string& group) const;

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

    virtual std::unordered_map<std::string, YAML::Node> metas(
        const std::string& group) const;

    virtual std::unordered_map<std::string, YAML::Node> metas(
        const std::string& group, const std::string& sensor_type
    ) const;

    template<typename T>
    cv::Mat createMat(const std::vector<size_t>& dims) const;

    std::shared_ptr<HighFive::File>  m_hdf5File;

    HDF5KernelConfig m_config;
};

using HDF5KernelPtr = std::shared_ptr<HDF5Kernel>;

} // namespace lvr2

#include "lvr2/io/kernels/HDF5Kernel.tcc"

#endif // HDF5KERNEL
