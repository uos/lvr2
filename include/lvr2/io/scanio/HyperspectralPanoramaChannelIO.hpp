#ifndef HYPERSPECTRALPANORAMACHANNELIO
#define HYPERSPECTRALPANORAMACHANNELIO

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/scanio/yaml/HyperspectralCamera.hpp"
#include "lvr2/io/baseio/MetaIO.hpp"
#include "lvr2/io/scanio/ImageIO.hpp"

using lvr2::baseio::MetaIO;
using lvr2::baseio::FeatureConstruct;

namespace lvr2 
{
namespace scanio
{

template <typename BaseIO>
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
    BaseIO *m_baseIO = static_cast<BaseIO*>(this);
    
    // deps
    MetaIO<BaseIO>* m_metaIO = static_cast<MetaIO<BaseIO>*>(m_baseIO);
    ImageIO<BaseIO>* m_imageIO = static_cast<ImageIO<BaseIO>*>(m_baseIO);
};

} // namespace scanio

/**
 *
 * @brief FeatureConstruct Specialization for HyperspectralPanoramaChannelIO
 * - Constructs dependencies (ImageIO)
 * - Sets type variable
 *
 */
template <typename T>
struct FeatureConstruct<lvr2::scanio::HyperspectralPanoramaChannelIO, T>
{
    // DEPS
    using dep1 = typename FeatureConstruct<lvr2::baseio::MetaIO, T>::type;
    using dep2 = typename FeatureConstruct<lvr2::scanio::ImageIO, T>::type;
    using deps = typename dep1::template Merge<dep2>;

    // ADD THE FEATURE ITSELF
    using type = typename deps::template add_features<lvr2::scanio::HyperspectralPanoramaChannelIO>::type;
};


} // namespace lvr2

#include "HyperspectralPanoramaChannelIO.tcc"

#endif // HYPERSPECTRALPANORAMACHANNELIO
