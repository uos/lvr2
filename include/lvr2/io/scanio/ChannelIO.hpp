#pragma once
#ifndef LVR2_IO_DESCRIPTIONS_CHANNELIO_HPP
#define LVR2_IO_DESCRIPTIONS_CHANNELIO_HPP

#include "lvr2/types/Channel.hpp"
#include "lvr2/util/Timestamp.hpp"

// Depending Features

namespace lvr2 {

namespace scanio
{

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

    template<typename T>
    void save(
        const size_t& scanPosNo,
        const size_t& lidarNo,
        const size_t& scanNo,
        const std::string& channelName,
        const Channel<T>& channel
    ) const;
    
    std::vector<size_t> loadDimensions(std::string groupName, std::string datasetName) const;

protected:
    
    // LOADER
    template<typename T>
    ChannelOptional<T> loadFundamental( 
        std::string group,
        std::string name) const;

    template<typename T>
    ChannelOptional<T> loadCustom(
        std::string group,
        std::string name) const;

    // SAVER

    template<typename T>
    void saveFundamental(
        std::string group,
        std::string name,
        const Channel<T>& channel) const;

    template<typename T>
    void saveCustom(
        std::string group,
        std::string name,
        const Channel<T>& channel) const;

    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);
};

} // namespace scanio

} // namespace lvr2

#include "ChannelIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_CHANNELIO_HPP