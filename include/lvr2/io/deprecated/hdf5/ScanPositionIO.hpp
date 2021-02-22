#pragma once

#ifndef LVR2_IO_HDF5_SCANPOSITIONIO_HPP
#define LVR2_IO_HDF5_SCANPOSITIONIO_HPP

#include "lvr2/types/ScanTypes.hpp"

#include <boost/optional.hpp>

// Dependencies
#include "ArrayIO.hpp"
#include "HyperspectralCameraIO.hpp"
#include "MatrixIO.hpp"
#include "ScanCameraIO.hpp"
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
    void save(uint scanPos, const ScanPositionPtr& scanPositionPtr);
    void save(HighFive::Group& group, const ScanPositionPtr& scanPositionPtr);

    ScanPositionPtr load(uint scanPos);
    ScanPositionPtr load(HighFive::Group& group);
    ScanPositionPtr loadScanPosition(uint scanPos);

  protected:
    bool isScanPosition(HighFive::Group& group);

    Derived* m_file_access = static_cast<Derived*>(this);
    // dependencies
    ArrayIO<Derived>* m_arrayIO = static_cast<ArrayIO<Derived>*>(m_file_access);
    MatrixIO<Derived>* m_matrixIO = static_cast<MatrixIO<Derived>*>(m_file_access);
    ScanIO<Derived>* m_scanIO = static_cast<ScanIO<Derived>*>(m_file_access);
    ScanCameraIO<Derived>* m_scanCameraIO = static_cast<ScanCameraIO<Derived>*>(m_file_access);
    HyperspectralCameraIO<Derived>* m_hyperspectralCameraIO =
        static_cast<HyperspectralCameraIO<Derived>*>(m_file_access);

    static constexpr const char* ID = "ScanPositionIO";
    static constexpr const char* OBJID = "ScanPosition";
};

} // namespace hdf5features

template <typename Derived>
struct Hdf5Construct<hdf5features::ScanPositionIO, Derived>
{

    // DEPS
    using dep1 = typename Hdf5Construct<hdf5features::ArrayIO, Derived>::type;
    using dep2 = typename Hdf5Construct<hdf5features::MatrixIO, Derived>::type;
    using dep3 = typename Hdf5Construct<hdf5features::ScanIO, Derived>::type;
    using dep4 = typename Hdf5Construct<hdf5features::ScanCameraIO, Derived>::type;
    using dep5 = typename Hdf5Construct<hdf5features::HyperspectralCameraIO, Derived>::type;
    using deps = typename dep1::template Merge<dep2>::template Merge<dep3>::template Merge<
        dep4>::template Merge<dep5>;

    // add the feature itself
    using type = typename deps::template add_features<hdf5features::ScanPositionIO>::type;
};

} // namespace lvr2

#include "ScanPositionIO.tcc"

#endif // LVR2_IO_HDF5_SCANPOSITIONIO_HPP
