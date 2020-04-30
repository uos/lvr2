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
template <typename FeatureBase>
class ScanProjectIO
{
  public:
    void saveScanProject(const ScanProjectPtr& scanProjectPtr);

    ScanProjectPtr loadScanProject();

  protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);
    // dependencies
    ScanPositionIO<FeatureBase>* m_scanPositionIO =
        static_cast<ScanPositionIO<FeatureBase>*>(m_featureBase);

    // static constexpr const char* ID = "ScanProjectIO";
    // static constexpr const char* OBJID = "ScanProject";
};

template <typename FeatureBase>
struct FeatureConstruct<ScanProjectIO, FeatureBase>
{

    // DEPS
    using deps = typename FeatureConstruct<ScanPositionIO, FeatureBase>::type;

    // add the feature itself
    using type = typename deps::template add_features<ScanProjectIO>::type;
};

} // namespace lvr2

#include "ScanProjectIO.tcc"

#endif // LVR2_IO_HDF5_SCANPROJECTIO_HPP
