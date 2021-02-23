
namespace lvr2
{

template <typename FeatureBase>
void HyperspectralCameraIO<FeatureBase>::save(
    const size_t& scanPosNo,
    const size_t& hCamNo,
    HyperspectralCameraPtr hcam) const
{
    auto Dgen = m_featureBase->m_description;

    Description d =  Dgen->hyperspectralCamera(scanPosNo, hCamNo);

    // std::cout << "[HypersprectralCameraIO - save]" << std::endl;
    // std::cout << d << std::endl;

    if(!d.dataRoot)
    {
        // someone doesnt want to save hyperspectral cameras
        return;
    }

    
    for(size_t i=0; i < hcam->panoramas.size(); i++)
    {
        m_hyperspectralPanoramaIO->save(scanPosNo, hCamNo, i, hcam->panoramas[i]);
    }


    // Save Meta
    if(d.meta)
    {
        YAML::Node meta;
        meta = *hcam;
        m_featureBase->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, meta);
    }
}

template <typename FeatureBase>
HyperspectralCameraPtr HyperspectralCameraIO<FeatureBase>::load(
        const size_t& scanPosNo,
        const size_t& hCamNo) const
{
    HyperspectralCameraPtr ret;

    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->hyperspectralCamera(scanPosNo, hCamNo);

    // std::cout << "[HypersprectralCameraIO - load]" << std::endl;
    // std::cout << d << std::endl;

    if(!d.dataRoot)
    {
        return ret;
    }

    if(!m_featureBase->m_kernel->exists(*d.dataRoot))
    {
        return ret;
    }

    // LOAD META
    if(d.meta)
    {
        YAML::Node meta;
        if(!m_featureBase->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
        {
            return ret;
        }
        ret = std::make_shared<HyperspectralCamera>(meta.as<HyperspectralCamera>());
    } else {
        ret.reset(new HyperspectralCamera);
    }

    // Load SensorData
    size_t hImageNo = 0;
    while(true)
    {
        HyperspectralPanoramaPtr pano = m_hyperspectralPanoramaIO->load(scanPosNo, hCamNo, hImageNo);
        if(pano)
        {
            ret->panoramas.push_back(pano);
        } else {
            break;
        }
        hImageNo++;
    }

    return ret;
}

template <typename FeatureBase>
boost::optional<YAML::Node> HyperspectralCameraIO<FeatureBase>::loadMeta(
    const size_t& scanPosNo,
    const size_t& hCamNo) const
{
    Description d = m_featureBase->m_description->hyperspectralCamera(scanPosNo, hCamNo);
    return m_metaIO->load(d);
}

} // namespace lvr2
