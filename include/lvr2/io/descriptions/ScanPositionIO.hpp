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
template <typename FeatureBase>
class ScanPositionIO
{
  public:
    void saveScanPosition(const size_t& scanPosNo, const ScanPositionPtr& scanPositionPtr);
    // void save(
    //   const std::string group&, 
    //   const ScanPositionPtr& scanPositionPtr);

    ScanPositionPtr loadScanPosition(const size_t& scanPosNo);
    //ScanPositionPtr load(const std::string& group, const std::string& container);

  protected:
    bool isScanPosition(const std::string& group);

    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);
    // dependencies
    ArrayIO<FeatureBase>* m_arrayIO = static_cast<ArrayIO<FeatureBase>*>(m_featureBase);
    MatrixIO<FeatureBase>* m_matrixIO = static_cast<MatrixIO<FeatureBase>*>(m_featureBase);
    ScanIO<FeatureBase>* m_scanIO = static_cast<ScanIO<FeatureBase>*>(m_featureBase);
    ScanCameraIO<FeatureBase>* m_scanCameraIO = static_cast<ScanCameraIO<FeatureBase>*>(m_featureBase);
    HyperspectralCameraIO<FeatureBase>* m_hyperspectralCameraIO =
        static_cast<HyperspectralCameraIO<FeatureBase>*>(m_featureBase);

    static constexpr const char* ID = "ScanPositionIO";
    static constexpr const char* OBJID = "ScanPosition";
};

template <typename FeatureBase>
struct FeatureConstruct< ScanPositionIO, FeatureBase>
{

    // DEPS
    using dep1 = typename FeatureConstruct<ArrayIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<MatrixIO, FeatureBase>::type;
    using dep3 = typename FeatureConstruct<ScanIO, FeatureBase>::type;
    using dep4 = typename FeatureConstruct<ScanCameraIO, FeatureBase>::type;
    using dep5 = typename FeatureConstruct<HyperspectralCameraIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>::template Merge<dep3>::template Merge<
        dep4>::template Merge<dep5>;

    // add the feature itself
    using type = typename deps::template add_features<ScanPositionIO>::type;
};

} // namespace lvr2

#include "ScanPositionIO.tcc"

#endif // LVR2_IO_HDF5_SCANPOSITIONIO_HPP
