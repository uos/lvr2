namespace lvr2 {

namespace scanio
{

template <typename Derived>
void HyperspectralPanoramaIO<Derived>::save(
    const size_t& scanPosNo, 
    const size_t& hCamNo, 
    const size_t& hPanoNo,
    HyperspectralPanoramaPtr pano) const
{
    auto Dgen = m_baseIO->m_description;
    Description d = Dgen->hyperspectralPanorama(scanPosNo, hCamNo, hPanoNo);

    // std::cout << "[HyperspectralPanoramaIO - save]" << std::endl;
    // std::cout << d << std::endl;

    if(!d.dataRoot)
    {
        d.dataRoot = "";
    }

    for(size_t i=0; i<pano->channels.size(); i++)
    {
        m_hyperspectralPanoramaChannelIO->save(scanPosNo, hCamNo, hPanoNo, i, pano->channels[i]);
    }

    // Save panorama preview
    if(!pano->preview.empty())
    {
        Description dp = Dgen->hyperspectralPanoramaPreview(scanPosNo, hCamNo, hPanoNo);
        m_imageIO->save(*dp.dataRoot, *dp.data, pano->preview);
    }

    if(d.meta)
    {
        YAML::Node node;
        node = *pano;
        m_baseIO->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
    }
    // store hyperspectral channels
}

template <typename Derived>
HyperspectralPanoramaPtr HyperspectralPanoramaIO<Derived>::load(
    const size_t& scanPosNo,
    const size_t& hCamNo,
    const size_t& hPanoNo) const
{
    HyperspectralPanoramaPtr ret;

    auto Dgen = m_baseIO->m_description;
    Description d = Dgen->hyperspectralPanorama(scanPosNo, hCamNo, hPanoNo);

    // std::cout << "[HyperspectralPanoramaIO - load]" << std::endl;
    // std::cout << d << std::endl;

    if(!d.dataRoot)
    {
        return ret;
    }

    // check if group exists
    if(!m_baseIO->m_kernel->exists(*d.dataRoot))
    {
        return ret;
    }

    /// META
    if(d.meta)
    {
        YAML::Node meta;
        if(!m_baseIO->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
        {
            return ret;
        }

        ret = std::make_shared<HyperspectralPanorama>(meta.as<HyperspectralPanorama>());
        ret->model = meta["camera_model"].as<CylindricalModel>();
    } else {
        
        // no meta name specified but scan position is there: 
        ret = std::make_shared<HyperspectralPanorama>();
    }

    /// Preview
    Description dp = Dgen->hyperspectralPanoramaPreview(scanPosNo, hCamNo, hPanoNo);
    auto preview = m_imageIO->load(*dp.dataRoot, *dp.data);
    if(preview)
    {
        ret->preview = *preview;
    }

    /// DATA

    // Load Hyperspectral Frame
    HyperspectralPanoramaChannelPtr hchannel = m_hyperspectralPanoramaChannelIO->load(scanPosNo, hCamNo, hPanoNo, 0);

    // hchannel contains matrix of whole spectral area 
    cv::Mat channels[hchannel->channel.channels()];

    // split hchannel into channels
    cv::split(hchannel->channel, channels);

    for(int i = 0; i < hchannel->channel.channels(); i++)
    {
        auto newChannel = std::make_shared<HyperspectralPanoramaChannel>();
        newChannel->channel = channels[i];
        ret->channels.push_back(newChannel);
    }
    ret->num_channels = hchannel->channel.channels();
    return ret;
}

template <typename Derived>
boost::optional<YAML::Node> HyperspectralPanoramaIO<Derived>::loadMeta(
    const size_t& scanPosNo,
    const size_t& hCamNo,
    const size_t& hPanoNo) const
{
    auto Dgen = m_baseIO->m_description;
    Description d = Dgen->hyperspectralPanorama(scanPosNo, hCamNo, hPanoNo);
    return m_metaIO->load(d);
}

} // namespace scanio

} // namespace lvr2