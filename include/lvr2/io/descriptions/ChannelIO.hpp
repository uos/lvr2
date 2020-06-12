#pragma once
#ifndef LVR2_IO_HDF5_CHANNELIO_HPP
#define LVR2_IO_HDF5_CHANNELIO_HPP

#include "lvr2/types/Channel.hpp"
#include "lvr2/io/GroupedChannelIO.hpp"
#include "lvr2/io/Timestamp.hpp"

// Depending Features

namespace lvr2 {

template<typename FeatureBase>
class ChannelIO 
{
public:
   
    UCharChannelOptional loadUCharChannel(std::string groupName, std::string datasetName);
    FloatChannelOptional loadFloatChannel(std::string groupName, std::string datasetName);
    DoubleChannelOptional loadDoubleChannel(std::string groupName, std::string datasetName);
    IndexChannelOptional loadIndexChannel(std::string groupName, std::string datasetName);
    
    void saveUCharChannel(std::string groupName, std::string datasetName, UCharChannel& channel);
    void saveFloatChannel(std::string groupName, std::string datasetName, FloatChannel& channel);
    void saveDoubleChannel(std::string groupName, std::string datasetName, DoubleChannel& channel);
    void saveIndexChannel(std::string groupName, std::string datasetName, UCharChannel& channel);

protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

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


} // namespace lvr2

#include "ChannelIO.tcc"

#endif