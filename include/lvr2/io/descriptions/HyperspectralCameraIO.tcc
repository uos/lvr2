#include "lvr2/io/yaml/HyperspectralCamera.hpp"

namespace lvr2
{

template <typename Derived>
void HyperspectralCameraIO<Derived>::save(
    const size_t& scanPosNo,
    const size_t& hCamNo,
    HyperspectralCameraPtr hcam) const
{
    auto d_gen = m_featureBase->m_description;

    Description d = d_gen->position(scanPosNo);
    d = d_gen->hyperspectralCamera(d, hCamNo);

    // std::cout << "[HypersprectralCameraIO - save]" << std::endl;
    // std::cout << d << std::endl;

    if(!d.groupName)
    {
        d.groupName = "";
    }

    
    for(size_t i=0; i<hcam->panoramas.size(); i++)
    {
        m_hyperspectralPanoramaIO->save(scanPosNo, hCamNo, i, hcam->panoramas[i]);
    }


    // Save Meta
    if(d.metaName)
    {
        YAML::Node meta;
        meta = *hcam;
        m_featureBase->m_kernel->saveMetaYAML(*d.groupName, *d.metaName, meta);
    }
}

template <typename Derived>
HyperspectralCameraPtr HyperspectralCameraIO<Derived>::load(
        const size_t& scanPosNo,
        const size_t& hCamNo) const
{
    HyperspectralCameraPtr ret;

    Description d_parent = m_featureBase->m_description->position(scanPosNo); 
    Description d = m_featureBase->m_description->hyperspectralCamera(d_parent, hCamNo);

    // std::cout << "[HypersprectralCameraIO - load]" << std::endl;
    // std::cout << d << std::endl;

    if(!d.groupName)
    {
        return ret;
    }

    if(!m_featureBase->m_kernel->exists(*d.groupName))
    {
        return ret;
    }

    if(d.metaName)
    {
        YAML::Node meta;
        m_featureBase->m_kernel->loadMetaYAML(*d.groupName, *d.metaName, meta);
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

} // namespace lvr2
