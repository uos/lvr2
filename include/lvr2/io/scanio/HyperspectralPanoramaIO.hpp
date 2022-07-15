#ifndef HYPERSPECTRALPANORAMAIO
#define HYPERSPECTRALPANORAMAIO

#include "lvr2/types/ScanTypes.hpp"


// deps
#include "lvr2/io/baseio/MetaIO.hpp"
#include "lvr2/io/scanio/yaml/HyperspectralCamera.hpp"
#include "lvr2/io/scanio/HyperspectralPanoramaChannelIO.hpp"

using lvr2::baseio::MetaIO;

namespace lvr2 
{
namespace scanio
{

template <typename BaseIO>
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
    BaseIO *m_baseIO = static_cast<BaseIO*>(this);

    // deps
    MetaIO<BaseIO>* m_metaIO = static_cast<MetaIO<BaseIO>*>(m_baseIO);
    ImageIO<BaseIO>* m_imageIO = static_cast<ImageIO<BaseIO>*>(m_baseIO); // for preview
    HyperspectralPanoramaChannelIO<BaseIO>* m_hyperspectralPanoramaChannelIO = static_cast<HyperspectralPanoramaChannelIO<BaseIO>*>(m_baseIO);
};

} // namespace scanio

template <typename T>
struct FeatureConstruct<lvr2::scanio::HyperspectralPanoramaIO, T>
{
    // DEPS
    using dep1 = typename FeatureConstruct<lvr2::baseio::MetaIO, T>::type;
    using dep2 = typename FeatureConstruct<lvr2::scanio::ImageIO, T>::type;
    using dep3 = typename FeatureConstruct<lvr2::scanio::HyperspectralPanoramaChannelIO, T>::type;
    using deps = typename dep1::template Merge<dep2>::template Merge<dep3>;
    
    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<lvr2::scanio::HyperspectralPanoramaIO>::type;
};

} // namespace lvr2

#include "HyperspectralPanoramaIO.tcc"

#endif // HYPERSPECTRALPANORAMAIO
