#pragma once

#ifndef LVR2_IO_HDF5_SCANPROJECTIO_HPP
#define LVR2_IO_HDF5_SCANPROJECTIO_HPP

#include "lvr2/types/ScanTypes.hpp"

#include <boost/optional.hpp>
#include <regex>

// Dependencies
#include "ArrayIO.hpp"
#include "MatrixIO.hpp"
#include "ScanIO.hpp"
#include "ScanPositionIO.hpp"

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
 * - IO: ScanProjectIO
 * - CLASS: ScanProject
 *
 * Dependencies:
 * - ScanPositionIO
 *
 */
template <typename Derived>
class ScanProjectIO
{
  public:
    void save(const ScanProjectPtr& scanProjectPtr);

    ScanProjectPtr load();

    ScanProjectPtr loadScanProject();

  protected:
    Derived* m_file_access = static_cast<Derived*>(this);
    // dependencies
    ScanPositionIO<Derived>* m_scanPositionIO =
        static_cast<ScanPositionIO<Derived>*>(m_file_access);

    // static constexpr const char* ID = "ScanProjectIO";
    // static constexpr const char* OBJID = "ScanProject";
};

} // namespace hdf5features

template <typename Derived>
struct Hdf5Construct<hdf5features::ScanProjectIO, Derived>
{

    // DEPS
    using deps = typename Hdf5Construct<hdf5features::ScanPositionIO, Derived>::type;

    // add the feature itself
    using type = typename deps::template add_features<hdf5features::ScanProjectIO>::type;
};

} // namespace lvr2

#include "ScanProjectIO.tcc"

#endif // LVR2_IO_HDF5_SCANPROJECTIO_HPP
