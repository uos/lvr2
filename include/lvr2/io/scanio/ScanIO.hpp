#pragma once

#ifndef LVR2_IO_DESCRIPTIONS_SCANIO_HPP
#define LVR2_IO_DESCRIPTIONS_SCANIO_HPP

#include <sstream>
#include <yaml-cpp/yaml.h>

#include "lvr2/io/yaml/Scan.hpp"
#include "lvr2/registration/OctreeReduction.hpp"
#include "lvr2/registration/ReductionAlgorithm.hpp"
#include "lvr2/util/Hdf5Util.hpp"
#include "lvr2/types/ScanTypes.hpp"

#include "MetaIO.hpp"
#include "PointCloudIO.hpp"

namespace lvr2
{

template <typename FeatureBase>
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

    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);

    // dependencies
    MetaIO<FeatureBase>* m_metaIO = static_cast<MetaIO<FeatureBase>*>(m_featureBase);
    PointCloudIO<FeatureBase>* m_pclIO = static_cast<PointCloudIO<FeatureBase>*>(m_featureBase);
    VariantChannelIO<FeatureBase>* m_vchannel_io = static_cast<VariantChannelIO<FeatureBase>*>(m_featureBase);

    static constexpr const char* ID = "ScanIO";
    static constexpr const char* OBJID = "Scan";
};

/**
 *
 * @brief FeatureConstruct Specialization for ScanIO
 * - Constructs dependencies (PointCloudIO, FullWaveformIO)
 * - Sets type variable
 *
 */
template <typename FeatureBase>
struct FeatureConstruct<ScanIO, FeatureBase >
{
    // DEPS
    using dep1 = typename FeatureConstruct<MetaIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<PointCloudIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<ScanIO>::type;
};

} // namespace lvr2

#include "ScanIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_SCANIO_HPP
