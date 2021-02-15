#include "lvr2/io/yaml/HyperspectralCamera.hpp"

namespace lvr2 {

template <typename Derived>
void HyperspectralPanoramaChannelIO<Derived>::save(
    const size_t& scanPosNo, 
    const size_t& hCamNo, 
    const size_t& hPanoNo,
    const size_t& channelId,
    HyperspectralPanoramaChannelPtr hchannel) const
{
    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->position(scanPosNo);
    d = Dgen->hyperspectralCamera(d, hCamNo);
    d = Dgen->hyperspectralPanorama(d, hPanoNo);
    d = Dgen->hyperspectralPanoramaChannel(d, channelId);

    // std::cout << "[HyperspectralPanoramaChannelIO - save]" << std::endl;
    // std::cout << d << std::endl;

    if(!d.groupName)
    {
        d.groupName = "";
    }

    // save image
    m_imageIO->save(*d.groupName, *d.dataSetName, hchannel->channel);

    // save meta
    if(d.metaName)
    {
        YAML::Node node;
        node = *hchannel;
        m_featureBase->m_kernel->saveMetaYAML(*d.groupName, *d.metaName, node);
    }
}

template <typename Derived>
HyperspectralPanoramaChannelPtr HyperspectralPanoramaChannelIO<Derived>::load(
    const size_t& scanPosNo,
    const size_t& hCamNo,
    const size_t& hPanoNo,
    const size_t& channelId) const
{
    HyperspectralPanoramaChannelPtr ret;

    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->position(scanPosNo);
    d = Dgen->hyperspectralCamera(d, hCamNo);
    d = Dgen->hyperspectralPanorama(d, hPanoNo);
    d = Dgen->hyperspectralPanoramaChannel(d, channelId);

    // std::cout << "[HyperspectralPanoramaChannelIO - load]" << std::endl;
    // std::cout << d << std::endl;

    if(!d.groupName)
    {
        d.groupName = "";
    }

    // check if group exists
    if(!m_featureBase->m_kernel->exists(*d.groupName))
    {
        return ret;
    }

    if(d.metaName)
    {
        if(!m_featureBase->m_kernel->exists(*d.groupName, *d.metaName))
        {
            return ret;
        } 

        YAML::Node meta;
        m_featureBase->m_kernel->loadMetaYAML(*d.groupName, *d.metaName, meta);
        ret = std::make_shared<HyperspectralPanoramaChannel>(meta.as<HyperspectralPanoramaChannel>());
    } else {
        
        // no meta name specified but scan position is there: 
        ret.reset(new HyperspectralPanoramaChannel);
    }
    
    ret->channel = *m_imageIO->load(*d.groupName, *d.dataSetName);

    return ret;
}

} // namespace lvr2