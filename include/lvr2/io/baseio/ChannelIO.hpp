#ifndef CHANNELIO
#define CHANNELIO

#include "lvr2/types/Channel.hpp"
#include "lvr2/util/Timestamp.hpp"

// Depending Features

namespace lvr2 
{
namespace baseio
{

template<typename BaseIO>
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

     BaseIO* m_baseIO = static_cast<BaseIO*>(this);
};

} // namespace baseio
} // namespace lvr2

#include "ChannelIO.tcc"

#endif // CHANNELIO
