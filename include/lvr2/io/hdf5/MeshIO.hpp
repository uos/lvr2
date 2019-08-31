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

/**
 * @class MeshIO 
 * @brief Hdf5IO Feature for handling MeshBuffer related IO
 * 
 * This Feature of the Hdf5IO handles the IO of a MeshBuffer object.
 * 
 * Example:
 * @code
 * MyHdf5IO io;
 * MeshBufferPtr mesh, mesh_in;
 * 
 * // writing
 * io.open("test.h5");
 * io.save("amesh", mesh);
 * 
 * // reading
 * mesh_in = io.loadMesh("amesh");
 * 
 * @endcode
 * 
 * Generates attributes at hdf5 group:
 * - IO: MeshIO
 * - CLASS: MeshBuffer
 * 
 * Dependencies:
 * - VariantChannelIO
 * 
 */
template<typename Derived>
class MeshIO {
public:
    void save(std::string name, const MeshBufferPtr& buffer);
    void save(HighFive::Group& group, const MeshBufferPtr& buffer);

    MeshBufferPtr load(std::string name);
    MeshBufferPtr load(HighFive::Group& group);
    MeshBufferPtr loadMesh(std::string name);

protected:

    bool isMesh(HighFive::Group& group);

    Derived* m_file_access = static_cast<Derived*>(this);
    // dependencies
    VariantChannelIO<Derived>* m_vchannel_io = static_cast<VariantChannelIO<Derived>*>(m_file_access);

    static constexpr char* ID = "MeshIO";
    static constexpr char* OBJID = "MeshBuffer";
};


} // hdf5features

} // namespace lvr2

#include "MeshIO.tcc"

#endif // LVR2_IO_HDF5_MESHIO_HPP
