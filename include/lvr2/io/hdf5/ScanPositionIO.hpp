#pragma once

#ifndef LVR2_IO_HDF5_SCANPOSITIONIO_HPP
#define LVR2_IO_HDF5_SCANPOSITIONIO_HPP

#include "lvr2/types/ScanTypes.hpp"

#include <boost/optional.hpp>

// Dependencies
#include "ArrayIO.hpp"
#include "MatrixIO.hpp"
#include "ScanIO.hpp"

namespace lvr2
{

namespace hdf5features
{

/**
 * @class ScanProjectIO
 * @brief Hdf5IO Feature for handling ScanProject related IO
 *
 * This Feature of the Hdf5IO handles the IO of a ScanProject object.
 *
 * Example:
 * @code
 * MyHdf5IO io;
 * PointBufferPtr pointcloud, pointcloud_in;
 *
 * // writing
 * io.open("test.h5");
 * io.save("apointcloud", pointcloud);
 *
 * // reading
 * pointcloud_in = io.loadPointCloud("apointcloud");
 *
 * @endcode
 *
 * Generates attributes at hdf5 group:
 * - IO: ScanPositionIO
 * - CLASS: ScanPosition
 *
 * Dependencies:
 * - ScanIO
 *
 */
template <typename Derived>
class ScanPositionIO
{
  public:
    void save(std::string name, const ScanPositionPtr& scanPositionPtr);

    void save(HighFive::Group& group, const ScanPositionPtr& scanPositionPtr);

    ScanPositionPtr load(HighFive::Group& group);

    ScanPositionPtr load(std::string name);

  protected:
    bool isScanPosition(HighFive::Group& group);

    Derived* m_file_access = static_cast<Derived*>(this);
    // dependencies
    ScanIO<Derived>* m_scanIO = static_cast<ScanIO<Derived>*>(m_file_access);

    static constexpr const char* ID = "ScanPositionIO";
    static constexpr const char* OBJID = "ScanPosition";
};

} // namespace hdf5features

template <typename Derived>
struct Hdf5Construct<hdf5features::ScanPositionIO, Derived>
{

    // DEPS
    using deps = typename Hdf5Construct<hdf5features::ScanIO, Derived>::type;

    // add the feature itself
    using type = typename deps::template add_features<hdf5features::ScanPositionIO>::type;
};

} // namespace lvr2

#include "ScanPositionIO.tcc"

#endif // LVR2_IO_HDF5_SCANPOSITIONIO_HPP
