#ifndef LVR2_IO_DESCRIPTIONS_HYPERSPECTRAL_PANORAMA_CHANNEL_IO_HPP
#define LVR2_IO_DESCRIPTIONS_HYPERSPECTRAL_PANORAMA_CHANNEL_IO_HPP

#include "lvr2/types/ScanTypes.hpp"

// deps
#include "MetaIO.hpp"
#include "ImageIO.hpp"


namespace lvr2 {

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

/**
 *
 * @brief FeatureConstruct Specialization for HyperspectralPanoramaChannelIO
 * - Constructs dependencies (ImageIO)
 * - Sets type variable
 *
 */
template <typename FeatureBase>
struct FeatureConstruct<HyperspectralPanoramaChannelIO, FeatureBase>
{
    // DEPS
    using dep1 = typename FeatureConstruct<MetaIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<ImageIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<HyperspectralPanoramaChannelIO>::type;
};

} // namespace lvr2

#include "HyperspectralPanoramaChannelIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_HYPERSPECTRAL_PANORAMA_CHANNEL_IO_HPP