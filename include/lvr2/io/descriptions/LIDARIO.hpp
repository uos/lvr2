#ifndef LVR2_IO_DESCRIPTIONS_LIDARIO_HPP
#define LVR2_IO_DESCRIPTIONS_LIDARIO_HPP

#include "ScanIO.hpp"

namespace lvr2
{

template <typename FeatureBase>
class LIDARIO
{
public:
    void save(const size_t& scanPosNo, const size_t& lidarNo, LIDARPtr lidar) const;

    LIDARPtr load(const size_t& scanPosNo, const size_t& lidarNo) const;

protected:
    FeatureBase* m_featureBase = static_cast<FeatureBase*>(this);
    
    // dependencies
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
    using deps = typename FeatureConstruct<ScanIO, FeatureBase>::type;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<LIDARIO>::type;
};

}

#include "LIDARIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_LIDARIO_HPP