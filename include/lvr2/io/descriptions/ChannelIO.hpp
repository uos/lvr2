#pragma once
#ifndef LVR2_IO_HDF5_CHANNELIO_HPP
#define LVR2_IO_HDF5_CHANNELIO_HPP

#include "lvr2/types/Channel.hpp"
#include "lvr2/io/Timestamp.hpp"

// Depending Features

namespace lvr2 {

template<typename FeatureBase>
class ChannelIO 
{
public:
   
    // base functionalities: save, load
    template<typename T> 
    ChannelOptional<T> load(
        std::string group,
        std::string name);

    template<typename T> 
    void save(
        std::string group,
        std::string name,
        const Channel<T>& channel);
    
    // convinience shortcuts
    UCharChannelOptional loadUCharChannel(std::string groupName, std::string datasetName);
    FloatChannelOptional loadFloatChannel(std::string groupName, std::string datasetName);
    DoubleChannelOptional loadDoubleChannel(std::string groupName, std::string datasetName);
    IndexChannelOptional loadIndexChannel(std::string groupName, std::string datasetName);
    
    void saveUCharChannel(std::string groupName, std::string datasetName, UCharChannel& channel);
    void saveFloatChannel(std::string groupName, std::string datasetName, FloatChannel& channel);
    void saveDoubleChannel(std::string groupName, std::string datasetName, DoubleChannel& channel);
    void saveIndexChannel(std::string groupName, std::string datasetName, UCharChannel& channel);

protected:

    // Need this functions:
    // - kernel is no template
    bool load(  std::string group,
                std::string name,
                Channel<float>& channel);

    bool load(  std::string group,
                std::string name,
                Channel<unsigned char>& channel);

    bool load(  std::string group,
                std::string name,
                Channel<double>& channel);

    bool load(  std::string group, 
                std::string name,
                Channel<int>& channel);

    bool load(  std::string group,
                std::string name,
                Channel<uint16_t>& channel);

    // TODO all save functions
    void _save( std::string group, 
                std::string name, 
                const Channel<float>& channel);

    void _save( std::string group,
                std::string name,
                const Channel<unsigned char>& channel);

    void _save( std::string group,
                std::string name,
                const Channel<double>& channel);

    void _save( std::string group,
                std::string name,
                const Channel<int>& channel);

    void _save( std::string group,
                std::string name,
                const Channel<uint16_t>& channel);

    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

};


} // namespace lvr2

#include "ChannelIO.tcc"

#endif