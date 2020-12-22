#pragma once
#ifndef LVR2_IO_DESCRIPTIONS_CHANNELIO_HPP
#define LVR2_IO_DESCRIPTIONS_CHANNELIO_HPP

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
        std::string name) const;

    template<typename T> 
    void save(
        std::string group,
        std::string name,
        const Channel<T>& channel) const;
    
    // convinience shortcuts
    UCharChannelOptional loadUCharChannel(std::string groupName, std::string datasetName);
    FloatChannelOptional loadFloatChannel(std::string groupName, std::string datasetName);
    DoubleChannelOptional loadDoubleChannel(std::string groupName, std::string datasetName);
    IndexChannelOptional loadIndexChannel(std::string groupName, std::string datasetName);
    
    // TODO:
    // void saveUCharChannel(std::string groupName, std::string datasetName, const UCharChannel& channel) const;
    // void saveFloatChannel(std::string groupName, std::string datasetName, const FloatChannel& channel) const;
    // void saveDoubleChannel(std::string groupName, std::string datasetName, const DoubleChannel& channel) const;
    // void saveIndexChannel(std::string groupName, std::string datasetName, const UCharChannel& channel) const;

    std::vector<size_t> loadDimensions(std::string groupName, std::string datasetName) const;

protected:

    // Need this functions:
    // - kernel is no template
    bool load(  std::string group,
                std::string name,
                Channel<float>& channel) const;

    bool load(  std::string group,
                std::string name,
                Channel<unsigned char>& channel) const;

    bool load(  std::string group,
                std::string name,
                Channel<double>& channel) const;

    bool load(  std::string group, 
                std::string name,
                Channel<int>& channel) const;

    bool load(  std::string group,
                std::string name,
                Channel<uint16_t>& channel) const;

    // TODO all save functions
    void _save( std::string group, 
                std::string name, 
                const Channel<float>& channel) const;

    void _save( std::string group,
                std::string name,
                const Channel<unsigned char>& channel) const;

    void _save( std::string group,
                std::string name,
                const Channel<double>& channel) const;

    void _save( std::string group,
                std::string name,
                const Channel<int>& channel) const;

    void _save( std::string group,
                std::string name,
                const Channel<uint16_t>& channel) const;

    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);
};


} // namespace lvr2

#include "ChannelIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_CHANNELIO_HPP