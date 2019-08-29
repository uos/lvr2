#pragma once

#ifndef LVR2_IO_HDF5_POINTBUFFERIO_HPP
#define LVR2_IO_HDF5_POINTBUFFERIO_HPP

#include <boost/optional.hpp>

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
    PointBufferPtr loadPointCloud(std::string name);

protected:

    Derived* m_file_access = static_cast<Derived*>(this);
    // dependencies
    VariantChannelIO<Derived>* m_vchannel_io = static_cast<VariantChannelIO<Derived>*>(m_file_access);
};


} // hdf5features

} // namespace lvr2 

#include "PointCloudIO.tcc"

#endif // LVR2_IO_HDF5_POINTBUFFERIO_HPP