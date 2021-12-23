namespace lvr2 {

template <typename Derived>
void HyperspectralPanoramaIO<Derived>::save(
    const size_t& scanPosNo, 
    const size_t& hCamNo, 
    const size_t& hPanoNo,
    HyperspectralPanoramaPtr pano) const
{
    auto Dgen = m_featureBase->m_description;
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
        m_featureBase->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
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

    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->hyperspectralPanorama(scanPosNo, hCamNo, hPanoNo);

    // std::cout << "[HyperspectralPanoramaIO - load]" << std::endl;
    // std::cout << d << std::endl;

    if(!d.dataRoot)
    {
        return ret;
    }

    // check if group exists
    if(!m_featureBase->m_kernel->exists(*d.dataRoot))
    {
        return ret;
    }

    /// META
    if(d.meta)
    {
        YAML::Node meta;
        if(!m_featureBase->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
        {
            return ret;
        }
        ret = std::make_shared<HyperspectralPanorama>(meta.as<HyperspectralPanorama>());
    } else {
        
        // no meta name specified but scan position is there: 
        ret.reset(new HyperspectralPanorama);
    }

    /// Preview
    Description dp = Dgen->hyperspectralPanoramaPreview(scanPosNo, hCamNo, hPanoNo);
    auto preview = m_imageIO->load(*dp.dataRoot, *dp.data);
    if(preview)
    {
        ret->preview = *preview;
    }

    /// DATA

    // Load Hyperspectral Channels
    size_t channelNo = 0;
    while(true)
    {
        HyperspectralPanoramaChannelPtr hchannel = m_hyperspectralPanoramaChannelIO->load(scanPosNo, hCamNo, hPanoNo, channelNo);
        if(hchannel)
        {
            ret->channels.push_back(hchannel);
        } else {
            break;
        }
        channelNo++;
    }


    return ret;
}

template <typename Derived>
boost::optional<YAML::Node> HyperspectralPanoramaIO<Derived>::loadMeta(
    const size_t& scanPosNo,
    const size_t& hCamNo,
    const size_t& hPanoNo) const
{
    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->hyperspectralPanorama(scanPosNo, hCamNo, hPanoNo);
    return m_metaIO->load(d);
}

} // namespace lvr2