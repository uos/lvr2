#ifndef LVR2_IO_DESCRIPTIONS_LIDARIO_HPP
#define LVR2_IO_DESCRIPTIONS_LIDARIO_HPP

#include "MetaIO.hpp"
#include "ScanIO.hpp"
#include "lvr2/io/scanio/yaml/LIDAR.hpp"

namespace lvr2
{

template <typename FeatureBase>
class LIDARIO
{
public:
    void save(const size_t& scanPosNo, const size_t& lidarNo, LIDARPtr lidar) const;

    boost::optional<YAML::Node> loadMeta(
        const size_t& scanPosNo,
        const size_t& lidarNo) const;

    LIDARPtr load(  const size_t& scanPosNo, 
                    const size_t& lidarNo) const;

protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);
    
    // dependencies
    MetaIO<FeatureBase>* m_metaIO = static_cast<MetaIO<FeatureBase>*>(m_featureBase);
    ScanIO<FeatureBase>* m_scanIO = static_cast<ScanIO<FeatureBase>*>(m_featureBase);
};

/**
 *
 * @brief FeatureConstruct Specialization for LIDARIO
 * - Constructs dependencies (PointCloudIO, FullWaveformIO)
 * - Sets type variable
 *
 */
template <typename FeatureBase>
struct FeatureConstruct<LIDARIO, FeatureBase >
{
    // DEPS
    using dep1 = typename FeatureConstruct<ScanIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<MetaIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<LIDARIO>::type;
};

}

#include "LIDARIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_LIDARIO_HPP