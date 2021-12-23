#pragma once

#ifndef LVR2_IO_DESCRIPTIONS_SCANPOSITIONIO_HPP
#define LVR2_IO_DESCRIPTIONS_SCANPOSITIONIO_HPP

#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>

#include "lvr2/io/scanio/yaml/ScanPosition.hpp"
#include "lvr2/types/ScanTypes.hpp"

// Dependencies
// SensorIOs
#include "HyperspectralCameraIO.hpp"
#include "CameraIO.hpp"
#include "LIDARIO.hpp"
#include "MetaIO.hpp"
// #include "LIDARIO.hpp"

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

    void save(
        const size_t& scanPosNo,
        ScanPositionPtr scanPositionPtr
        ) const;

    boost::optional<YAML::Node> loadMeta(
        const size_t& scanPosNo) const;

    ScanPositionPtr load(
        const size_t& scanPosNo) const;

    ScanPositionPtr load(
        const size_t& scanPosNo, 
        ReductionAlgorithmPtr reduction) const;

    void saveScanPosition(
        const size_t& scanPosNo,
        ScanPositionPtr scanPositionPtr
        ) const;

    ScanPositionPtr loadScanPosition(
        const size_t& scanPosNo) const;

    ScanPositionPtr loadScanPosition(
        const size_t& scanPosNo, 
        ReductionAlgorithmPtr reduction) const;
   
  protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);
    
    // dependencies
    MetaIO<FeatureBase>* m_metaIO = static_cast<MetaIO<FeatureBase>*>(m_featureBase);
    LIDARIO<FeatureBase>* m_lidarIO = static_cast<LIDARIO<FeatureBase>*>(m_featureBase);
    CameraIO<FeatureBase>* m_cameraIO = static_cast<CameraIO<FeatureBase>*>(m_featureBase);
    HyperspectralCameraIO<FeatureBase>* m_hyperspectralCameraIO = static_cast<HyperspectralCameraIO<FeatureBase>*>(m_featureBase);

    static constexpr const char* ID = "ScanPositionIO";
    static constexpr const char* OBJID = "ScanPosition";
};

template <typename FeatureBase>
struct FeatureConstruct< ScanPositionIO, FeatureBase>
{
    // DEPS
    using dep1 = typename FeatureConstruct<LIDARIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<CameraIO, FeatureBase>::type;
    using dep3 = typename FeatureConstruct<HyperspectralCameraIO, FeatureBase>::type;
    using dep4 = typename FeatureConstruct<MetaIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>::template Merge<dep3>::template Merge<dep4>;

    // add the feature itself
    using type = typename deps::template add_features<ScanPositionIO>::type;
};

} // namespace lvr2

#include "ScanPositionIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_SCANPOSITIONIO_HPP