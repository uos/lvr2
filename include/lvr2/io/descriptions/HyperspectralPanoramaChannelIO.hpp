#ifndef LVR2_IO_DESCRIPTIONS_HYPERSPECTRAL_PANORAMA_CHANNEL_IO_HPP
#define LVR2_IO_DESCRIPTIONS_HYPERSPECTRAL_PANORAMA_CHANNEL_IO_HPP

#include "lvr2/types/ScanTypes.hpp"
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

protected:
    FeatureBase *m_featureBase = static_cast<FeatureBase*>(this);
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
    using deps = typename FeatureConstruct<ImageIO, FeatureBase>::type;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<HyperspectralPanoramaChannelIO>::type;
};

} // namespace lvr2

#include "HyperspectralPanoramaChannelIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_HYPERSPECTRAL_PANORAMA_CHANNEL_IO_HPP