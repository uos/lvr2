#pragma once
#ifndef LVR2_IO_HDF5_CHANNELIO_HPP
#define LVR2_IO_HDF5_CHANNELIO_HPP

#include "lvr2/types/Channel.hpp"

// Depending Features

namespace lvr2 {

namespace hdf5features {

template<typename Derived>
class ChannelIO {
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

};


} // namespace hdf5features

} // namespace hdf5features

#include "ChannelIO.tcc"

#endif