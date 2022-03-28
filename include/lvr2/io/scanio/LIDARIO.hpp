#ifndef LIDARIO_HPP
#define LIDARIO_HPP

#include "lvr2/io/baseio/MetaIO.hpp"
#include "lvr2/io/scanio/ScanIO.hpp"
#include "lvr2/io/scanio/yaml/LIDAR.hpp"

namespace lvr2
{
namespace scanio
{

template <typename BaseIO>
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
    BaseIO* m_baseIO = static_cast<BaseIO*>(this);
    
    // dependencies
    MetaIO<BaseIO>* m_metaIO = static_cast<MetaIO<BaseIO>*>(m_baseIO);
    ScanIO<BaseIO>* m_scanIO = static_cast<ScanIO<BaseIO>*>(m_baseIO);
};

} // namespace scanio

/**
 *
 * @brief FeatureConstruct Specialization for LIDARIO
 * - Constructs dependencies (PointCloudIO, FullWaveformIO)
 * - Sets type variable
 *
 */
template <typename T>
struct FeatureConstruct<lvr2::scanio::LIDARIO, T >
{
    // DEPS
    using dep1 = typename FeatureConstruct<lvr2::scanio::ScanIO, T>::type;
    using dep2 = typename FeatureConstruct<lvr2::baseio::MetaIO, T>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<lvr2::scanio::LIDARIO>::type;
};

} // namespace lvr2

#include "LIDARIO.tcc"

#endif // LIDARIO
