#ifndef LVR2_IO_DESCRIPTIONS_HYPERSPECTRAL_PANORAMA_IO_HPP
#define LVR2_IO_DESCRIPTIONS_HYPERSPECTRAL_PANORAMA_IO_HPP

#include "lvr2/types/ScanTypes.hpp"


// deps
#include "MetaIO.hpp"
#include "lvr2/io/scanio/yaml/HyperspectralCamera.hpp"
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

    boost::optional<YAML::Node> loadMeta(
        const size_t& scanPosNo,
        const size_t& hCamNo,
        const size_t& hPanoNo) const;

protected:
    FeatureBase *m_featureBase = static_cast<FeatureBase*>(this);

    // deps
    MetaIO<FeatureBase>* m_metaIO = static_cast<MetaIO<FeatureBase>*>(m_featureBase);
    ImageIO<FeatureBase>* m_imageIO = static_cast<ImageIO<FeatureBase>*>(m_featureBase); // for preview
    HyperspectralPanoramaChannelIO<FeatureBase>* m_hyperspectralPanoramaChannelIO = static_cast<HyperspectralPanoramaChannelIO<FeatureBase>*>(m_featureBase);
};

template <typename FeatureBase>
struct FeatureConstruct<HyperspectralPanoramaIO, FeatureBase>
{
    // DEPS
    using dep1 = typename FeatureConstruct<MetaIO, FeatureBase>::type;
    using dep2 = typename FeatureConstruct<ImageIO, FeatureBase>::type;
    using dep3 = typename FeatureConstruct<HyperspectralPanoramaChannelIO, FeatureBase>::type;
    using deps = typename dep1::template Merge<dep2>::template Merge<dep3>;
    
    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<HyperspectralPanoramaIO>::type;
};

} // namespace lvr2

#include "HyperspectralPanoramaIO.tcc"

#endif // LVR2_IO_DESCRIPTIONS_HYPERSPECTRAL_PANORAMA_IO_HPP