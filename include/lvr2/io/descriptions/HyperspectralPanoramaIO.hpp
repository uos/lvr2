#ifndef LVR2_IO_DESCRIPTIONS_HYPERSPECTRAL_PANORAMA_IO_HPP
#define LVR2_IO_DESCRIPTIONS_HYPERSPECTRAL_PANORAMA_IO_HPP

#include "lvr2/types/ScanTypes.hpp"
#include "HyperspectralPanoramaChannelIO.hpp"

namespace lvr2 {

template <typename FeatureBase>
class HyperspectralPanoramaIO
{
public:
    void save(
        const size_t& scanPosNo, 
        const size_t& hCamNo, 
        const size_t& hPanoNo,
        HyperspectralPanoramaPtr hcam) const;

    HyperspectralPanoramaPtr load(
        const size_t& scanPosNo,
        const size_t& hCamNo,
        const size_t& hPanoNo) const;

protected:
    FeatureBase *m_featureBase = static_cast<FeatureBase*>(this);
    HyperspectralPanoramaChannelIO<FeatureBase>* m_hyperspectralPanoramaChannelIO = static_cast<HyperspectralPanoramaChannelIO<FeatureBase>*>(m_featureBase);
};

template <typename FeatureBase>
struct FeatureConstruct<HyperspectralPanoramaIO, FeatureBase>
{
    // DEPS
    using deps = typename FeatureConstruct<HyperspectralPanoramaChannelIO, FeatureBase>::type;
    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<HyperspectralPanoramaIO>::type;
};

} // namespace lvr2

#include "HyperspectralPanoramaIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_HYPERSPECTRAL_PANORAMA_IO_HPP