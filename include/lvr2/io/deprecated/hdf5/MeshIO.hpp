#pragma once

#ifndef LVR2_IO_HDF5_MESHIO_HPP
#define LVR2_IO_HDF5_MESHIO_HPP

#include <boost/optional.hpp>

#include "lvr2/io/MeshBuffer.hpp"
#include "lvr2/io/AttributeMeshIOBase.hpp"
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

// Dependencies
#include "ChannelIO.hpp"
#include "VariantChannelIO.hpp"
#include "ArrayIO.hpp"

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
 * io.save("ames#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"h", mesh);
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
 * - ArrayIO
 * 
 */
template<typename Derived>
class MeshIO : public AttributeMeshIOBase {
public:
    void save(std::string name, const MeshBufferPtr& buffer);
    void save(HighFive::Group& group, const MeshBufferPtr& buffer);

    MeshBufferPtr load(std::string name);
    MeshBufferPtr load(HighFive::Group& group);
    MeshBufferPtr loadMesh(std::string name);

    void setMeshName(std::string meshName);

protected:

    bool isMesh(HighFive::Group& group);

    Derived* m_file_access = static_cast<Derived*>(this);

    std::string m_mesh_name = "";

    // dependencies
    VariantChannelIO<Derived>* m_vchannel_io = static_cast<VariantChannelIO<Derived>*>(m_file_access);

    static constexpr const char* ID = "MeshIO";
    static constexpr const char* OBJID = "MeshBuffer";

    ArrayIO<Derived>* m_array_io = static_cast<ArrayIO<Derived>*>(m_file_access);

    /**
     * @brief Persistence layer interface, Accesses the vertices of the mesh in the persistence layer.
     * @return An optional float channel, the channel is valid if the mesh vertices have been read successfully
     */
    FloatChannelOptional getVertices();

    /**
     * @brief Persistence layer interface, Accesses the face indices of the mesh in the persistence layer.
     * @return An optional index channel, the channel is valid if the mesh indices have been read successfully
     */
    IndexChannelOptional getIndices();

    /**
     * @brief Persistence layer interface, Writes the vertices of the mesh to the persistence layer.
     * @return true if the channel has been written successfully
     */
    bool addVertices(const FloatChannel& channel_ptr);

    /**
     * @brief Persistence layer interface, Writes the face indices of the mesh to the persistence layer.
     * @return true if the channel has been written successfully
     */
    bool addIndices(const IndexChannel& channel_ptr);

    template <typename T>
    bool getChannel(const std::string group, const std::string name, boost::optional<AttributeChannel<T>>& channel);

    /**
     * @brief getChannel  Reads a float attribute channel in the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the float channel
     * @return            true if the channel has been loaded successfully, false otherwise
     */
    virtual bool getChannel(const std::string group, const std::string name, FloatChannelOptional& channel);

    /**
     * @brief getChannel  Reads an index attribute channel in the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the index channel
     * @return            true if the channel has been loaded successfully, false otherwise
     */
    virtual bool getChannel(const std::string group, const std::string name, IndexChannelOptional& channel);

    /**
     * @brief getChannel  Reads an unsigned char attribute channel in the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the unsigned char channel
     * @return            true if the channel has been loaded successfully, false otherwise
     */
    virtual bool getChannel(const std::string group, const std::string name, UCharChannelOptional& channel);

    template <typename T>
    bool addChannel(const std::string group, const std::string name, const AttributeChannel<T>& channel);

    /**
     * @brief addChannel  Writes a float attribute channel from the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the float channel which should be written
     * @return            true if the channel has been written successfully, false otherwise
     */
    virtual bool addChannel(const std::string group, const std::string name, const FloatChannel& channel);

    /**
     * @brief addChannel  Writes an index attribute channel from the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the index channel which should be written
     * @return            true if the channel has been written successfully, false otherwise
     */
    virtual bool addChannel(const std::string group, const std::string name, const IndexChannel& channel);

    /**
     * @brief addChannel  Writes an unsigned char attribute channel from the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the unsigned char channel which should be written
     * @return            true if the channel has been written successfully, false otherwise
     */
    virtual bool addChannel(const std::string group, const std::string name, const UCharChannel& channel);


};


} // hdf5features


/** 
 * 
 * @brief Hdf5Construct Specialization for hdf5features::MeshIO
 * - Constructs dependencies (VariantChannelIO, ArrayIO)
 * - Sets type variable
 * 
 */
template<typename Derived>
struct Hdf5Construct<hdf5features::MeshIO, Derived> {
    
    // DEPS
    using dep1 = typename Hdf5Construct<hdf5features::VariantChannelIO, Derived>::type;
    using dep2 = typename Hdf5Construct<hdf5features::ArrayIO, Derived>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<hdf5features::MeshIO>::type;
     
};

} // namespace lvr2

#include "MeshIO.tcc"

#endif // LVR2_IO_HDF5_MESHIO_HPP
