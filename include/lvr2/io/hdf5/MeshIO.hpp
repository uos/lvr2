#pragma once

#ifndef LVR2_IO_HDF5_MESHIO_HPP
#define LVR2_IO_HDF5_MESHIO_HPP

#include <boost/optional.hpp>

#include "lvr2/io/MeshBuffer.hpp"

// Dependencies
#include "ChannelIO.hpp"
#include "VariantChannelIO.hpp"

namespace lvr2 {

namespace hdf5features {

template<typename Derived>
class MeshIO {
public:
    void save(std::string name, const MeshBufferPtr& buffer);
    void save(HighFive::Group& group, const MeshBufferPtr& buffer);

    MeshBufferPtr load(std::string name);
    MeshBufferPtr load(HighFive::Group& group);
    MeshBufferPtr loadMesh(std::string name);

protected:

    Derived* m_file_access = static_cast<Derived*>(this);
    // dependencies
    VariantChannelIO<Derived>* m_vchannel_io = static_cast<VariantChannelIO<Derived>*>(m_file_access);
};


} // hdf5features

} // namespace lvr2

#include "MeshIO.tcc"

#endif // LVR2_IO_HDF5_MESHIO_HPP
