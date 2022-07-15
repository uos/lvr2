#pragma once

#ifndef SCANIO
#define SCANIO

#include <sstream>
#include <yaml-cpp/yaml.h>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/baseio/MetaIO.hpp"
#include "lvr2/io/scanio/PointCloudIO.hpp"
#include "lvr2/io/scanio/yaml/Scan.hpp"
#include "lvr2/registration/OctreeReduction.hpp"
#include "lvr2/registration/ReductionAlgorithm.hpp"
#include "lvr2/util/Hdf5Util.hpp"

namespace lvr2
{
namespace scanio
{

template <typename BaseIO>
class ScanIO
{
  public:
  
    void save(
            const size_t& scanPosNo, 
            const size_t& sensorNo,
            const size_t& scanNo,
            ScanPtr buffer
        ) const;

    boost::optional<YAML::Node> loadMeta(
        const size_t& scanPosNo, 
        const size_t& sensorNo,
        const size_t& scanNo
    ) const;

    ScanPtr load(
        const size_t& scanPosNo, 
        const size_t& sensorNo,
        const size_t& scanNo
    ) const;

    std::unordered_map<std::string, YAML::Node> loadChannelMetas(
        const size_t& scanPosNo, 
        const size_t& sensorNo,
        const size_t& scanNo
    ) const;

    void saveScan(
            const size_t& scanPosNo, 
            const size_t& sensorNo,
            const size_t& scanNo, 
            ScanPtr buffer
        ) const;
  
    ScanPtr loadScan(
            const size_t& scanPosNo, 
            const size_t& sensorNo,
            const size_t& scanNo
        ) const;

    ScanPtr loadScan(
            const size_t& scanPosNo, 
            const size_t& sensorNo,
            const size_t& scanNo,
            ReductionAlgorithmPtr reduction
        ) const;

  protected:

    BaseIO* m_baseIO = static_cast<BaseIO*>(this);

    // dependencies
    MetaIO<BaseIO>* m_metaIO = static_cast<MetaIO<BaseIO>*>(m_baseIO);
    PointCloudIO<BaseIO>* m_pclIO = static_cast<PointCloudIO<BaseIO>*>(m_baseIO);
    VariantChannelIO<BaseIO>* m_vchannel_io = static_cast<VariantChannelIO<BaseIO>*>(m_baseIO);

    static constexpr const char* ID = "ScanIO";
    static constexpr const char* OBJID = "Scan";
};

} // namespace scanio

/**
 *
 * @brief FeatureConstruct Specialization for ScanIO
 * - Constructs dependencies (PointCloudIO, FullWaveformIO)
 * - Sets type variable
 *
 */
template <typename T>
struct FeatureConstruct<lvr2::scanio::ScanIO, T>
{
    // DEPS
    using dep1 = typename FeatureConstruct<lvr2::baseio::MetaIO, T>::type;
    using dep2 = typename FeatureConstruct<lvr2::scanio::PointCloudIO, T>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<lvr2::scanio::ScanIO>::type;
};

} // namespace lvr2

#include "ScanIO.tcc"

#endif // SCANIO
