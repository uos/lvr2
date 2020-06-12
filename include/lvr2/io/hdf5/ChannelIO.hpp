#pragma once
#ifndef LVR2_IO_HDF5_CHANNELIO_HPP
#define LVR2_IO_HDF5_CHANNELIO_HPP

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

#include "lvr2/types/Channel.hpp"
#include "lvr2/io/GroupedChannelIO.hpp"
#include "lvr2/io/Timestamp.hpp"

// Depending Features

namespace lvr2 {

namespace hdf5features {

template<typename Derived>
class ChannelIO : public GroupedChannelIO {
public:
    template<typename T>
    ChannelOptional<T> load(std::string groupName,
        std::string datasetName);

    template<typename T>
    ChannelOptional<T> load(
        HighFive::Group& g,
        std::string datasetName
    );

    template<typename T>
    ChannelOptional<T> loadChannel(std::string groupName,
        std::string datasetName);

    template<typename T>
    void save(std::string groupName,
        std::string datasetName,
        const Channel<T>& channel);

    template<typename T>
    void save(HighFive::Group& g,
        std::string datasetName,
        const Channel<T>& channel);

    template<typename T>
    void save(HighFive::Group& g,
        std::string datasetName,
        const Channel<T>& channel,
        std::vector<hsize_t>& chunkSize);

protected:
    Derived* m_file_access = static_cast<Derived*>(this);

    template <typename T>
    bool getChannel(const std::string group, const std::string name, boost::optional<AttributeChannel<T>>& channel);

    /**
     * @brief getChannel  Reads a float attribute channel in the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the float channel
     * @return            true if the channel has been loaded successfully, false otherwise
     */
    virtual bool getChannel(const std::string group, const std::string name, FloatChannelOptional& channel);

    /**
     * @brief getChannel  Reads an index attribute channel in the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the index channel
     * @return            true if the channel has been loaded successfully, false otherwise
     */
    virtual bool getChannel(const std::string group, const std::string name, IndexChannelOptional& channel);

    /**
     * @brief getChannel  Reads an unsigned char attribute channel in the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the unsigned char channel
     * @return            true if the channel has been loaded successfully, false otherwise
     */
    virtual bool getChannel(const std::string group, const std::string name, UCharChannelOptional& channel);

    template <typename T>
    bool addChannel(const std::string group, const std::string name, const AttributeChannel<T>& channel);

    /**
     * @brief addChannel  Writes a float attribute channel from the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the float channel which should be written
     * @return            true if the channel has been written successfully, false otherwise
     */
    virtual bool addChannel(const std::string group, const std::string name, const FloatChannel& channel);

    /**
     * @brief addChannel  Writes an index attribute channel from the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the index channel which should be written
     * @return            true if the channel has been written successfully, false otherwise
     */
    virtual bool addChannel(const std::string group, const std::string name, const IndexChannel& channel);

    /**
     * @brief addChannel  Writes an unsigned char attribute channel from the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the unsigned char channel which should be written
     * @return            true if the channel has been written successfully, false otherwise
     */
    virtual bool addChannel(const std::string group, const std::string name, const UCharChannel& channel);


};


} // namespace hdf5features

} // namespace hdf5features

#include "ChannelIO.tcc"

#endif