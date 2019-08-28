#pragma once

#ifndef LVR2_IO_HDF5_POINTBUFFERIO_HPP
#define LVR2_IO_HDF5_POINTBUFFERIO_HPP

#include "lvr2/io/PointBuffer.hpp"

// Dependencies
#include "ChannelIO.hpp"
#include "VariantChannelIO.hpp"

namespace lvr2 {

namespace hdf5features {

template<typename Derived>
class PointCloudIO {
public:
    void save(std::string name, const PointBufferPtr& buffer);
    void save(HighFive::Group& group, const PointBufferPtr& buffer);

    PointBufferPtr load(std::string name);
    PointBufferPtr load(HighFive::Group& group);

protected:

    PointBuffer::val_type loadDynamic(
        HighFive::DataType dtype,
        HighFive::Group& group,
        std::string name
    );
    

    Derived* m_file_access = static_cast<Derived*>(this);
    ChannelIO<Derived>* m_channel_io = static_cast<ChannelIO<Derived>*>(m_file_access);
};


} // hdf5features

} // namespace lvr2 

#include "PointCloudIO.tcc"

#endif // LVR2_IO_HDF5_POINTBUFFERIO_HPP