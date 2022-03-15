#ifndef HYPERSPECTRALPANORAMACHANNELIO
#define HYPERSPECTRALPANORAMACHANNELIO

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/scanio/yaml/HyperspectralCamera.hpp"
// deps
#include "MetaIO.hpp"
#include "ImageIO.hpp"


namespace lvr2 
{

namespace scanio
{

template <typename FeatureBase>
class HyperspectralPanoramaChannelIO
{
public:
    void save(
        const size_t& scanPosNo,
        const size_t& hCamNo, 
        const size_t& hPanoNo,
        const size_t& channelId,
        HyperspectralPanoramaChannelPtr hcam) const;

    HyperspectralPanoramaChannelPtr load(
        const size_t& scanPosNo,
        const size_t& hCamNo,
        const size_t& hPanoNo,
        const size_t& channelId) const;

    boost::optional<YAML::Node> loadMeta(
        const size_t& scanPosNo,
        const size_t& hCamNo,
        const size_t& hPanoNo,
        const size_t& channelId) const;

protected:
    FeatureBase *m_featureBase = static_cast<FeatureBase*>(this);
    
    // deps
    MetaIO<FeatureBase>* m_metaIO = static_cast<MetaIO<FeatureBase>*>(m_featureBase);
    ImageIO<FeatureBase>* m_imageIO = static_cast<ImageIO<FeatureBase>*>(m_featureBase);
};

} // namespace scanio

/**
 *
 * @brief FeatureConstruct Specialization for HyperspectralPanoramaChannelIO
 * - Constructs dependencies (ImageIO)
 * - Sets type variable
 *
 */
template <typename FeatureBase>
struct FeatureConstruct<lvr2::scanio::HyperspectralPanoramaChannelIO, FeatureBase>
{
    // DEPS
    using dep1 = typename FeatureConstruct<lvr2::scanio::MetaIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<lvr2::scanio::ImageIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<lvr2::scanio::HyperspectralPanoramaChannelIO>::type;
};


} // namespace lvr2

#include "HyperspectralPanoramaChannelIO.tcc"

#endif // HYPERSPECTRALPANORAMACHANNELIO
