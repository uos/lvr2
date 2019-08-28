#pragma once
#ifndef LVR2_IO_HDF5_VARIANTCHANNELIO_HPP
#define LVR2_IO_HDF5_VARIANTCHANNELIO_HPP

#include "lvr2/types/VariantChannel.hpp"

namespace lvr2 {

namespace hdf5features {

template<typename Derived>
class VariantChannelIO {
public:

    template<typename ...Tp>
    void save(std::string groupName, std::string datasetName, const VariantChannel<Tp...>& vchannel);
    
    template<typename ...Tp>
    void save(HighFive::Group& group, std::string datasetName, const VariantChannel<Tp...>& vchannel);
    
    template<typename ...Tp>
    VariantChannelOptional<Tp...> load(std::string groupName, std::string datasetName);
    
    template<typename ...Tp>
    VariantChannelOptional<Tp...> load(HighFive::Group& group, std::string datasetName);
    
protected:

    template<typename ...Tp>
    VariantChannelOptional<Tp...> loadDynamic(HighFive::DataType dtype,
        HighFive::Group& group,
        std::string name);

    template<typename ...Tp>
    void saveDynamic(HighFive::Group& group,
        std::string datasetName,
        const VariantChannel<Tp...>& vchannel
    );

    Derived* m_file_access = static_cast<Derived*>(this);
    ChannelIO<Derived>* m_channel_io = static_cast<ChannelIO<Derived>*>(m_file_access);
};


} // hdf5features

} // namespace lvr2 

#include "VariantChannelIO.tcc"

#endif // LVR2_IO_HDF5_VARIANTCHANNELIO_HPP