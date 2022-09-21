#pragma once

#ifndef SCANPOSITIONIO
#define SCANPOSITIONIO

#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/baseio/MetaIO.hpp"
#include "lvr2/io/scanio/yaml/ScanPosition.hpp"
#include "lvr2/io/scanio/HyperspectralCameraIO.hpp"
#include "lvr2/io/scanio/CameraIO.hpp"
#include "lvr2/io/scanio/LIDARIO.hpp"

namespace lvr2
{
namespace scanio
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
template <typename BaseIO>
class ScanPositionIO
{
  public:

    bool save(
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

    bool saveScanPosition(
        const size_t& scanPosNo,
        ScanPositionPtr scanPositionPtr
        ) const;

    ScanPositionPtr loadScanPosition(
        const size_t& scanPosNo) const;

    ScanPositionPtr loadScanPosition(
        const size_t& scanPosNo, 
        ReductionAlgorithmPtr reduction) const;
   
  protected:
    BaseIO* m_baseIO = static_cast<BaseIO*>(this);
    
    // dependencies
    MetaIO<BaseIO>* m_metaIO = static_cast<MetaIO<BaseIO>*>(m_baseIO);
    LIDARIO<BaseIO>* m_lidarIO = static_cast<LIDARIO<BaseIO>*>(m_baseIO);
    CameraIO<BaseIO>* m_cameraIO = static_cast<CameraIO<BaseIO>*>(m_baseIO);
    HyperspectralCameraIO<BaseIO>* m_hyperspectralCameraIO = static_cast<HyperspectralCameraIO<BaseIO>*>(m_baseIO);

    static constexpr const char* ID = "ScanPositionIO";
    static constexpr const char* OBJID = "ScanPosition";
};

} // namespace scanio

template <typename T>
struct FeatureConstruct<lvr2::scanio::ScanPositionIO, T>
{
    // DEPS
    using dep1 = typename FeatureConstruct<lvr2::scanio::LIDARIO, T>::type;
    using dep2 = typename FeatureConstruct<lvr2::scanio::CameraIO, T>::type;
    using dep3 = typename FeatureConstruct<lvr2::scanio::HyperspectralCameraIO, T>::type;
    using dep4 = typename FeatureConstruct<lvr2::baseio::MetaIO, T>::type;
    using deps = typename dep1::template Merge<dep2>::template Merge<dep3>::template Merge<dep4>;

    // add the feature itself
    using type = typename deps::template add_features<lvr2::scanio::ScanPositionIO>::type;
};

} // namespace lvr2

#include "ScanPositionIO.tcc"

#endif // SCANPOSITIONIO
