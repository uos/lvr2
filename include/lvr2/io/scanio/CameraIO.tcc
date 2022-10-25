namespace lvr2
{

namespace scanio
{

template <typename BaseIO>
void CameraIO<BaseIO>::save(
    const size_t& scanPosNo,
    const size_t& scanCamNo,
    CameraPtr cameraPtr) const
{
    auto Dgen = m_baseIO->m_description;

    Description d = Dgen->camera(scanPosNo, scanCamNo);

    if(!d.dataRoot)
    {
        // someone doesnt want to save cameras
        return;
    }

    for(size_t i = 0; i < cameraPtr->groups.size(); i++)
    {
        m_cameraImageGroupIO->save(scanPosNo, scanCamNo, i, cameraPtr->groups[i]);
    }

    // writing meta
    if(d.meta)
    {
        YAML::Node node;
        node = *cameraPtr;
        m_baseIO->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
    }
}

template <typename BaseIO>
CameraPtr CameraIO<BaseIO>::load(
    const size_t& scanPosNo, 
    const size_t& scanCamNo) const
{
    CameraPtr ret;

    auto Dgen = m_baseIO->m_description;
    Description d = Dgen->camera(scanPosNo, scanCamNo);

    if(!d.dataRoot)
    {
        std::cout << "Data root is not set" << std::endl;
        return ret;
    }

    if(!m_baseIO->m_kernel->exists(*d.dataRoot))
    {
        std::cout << "Data root does not exist" << std::endl;
        return ret;
    }

    /// META
    if (d.meta)
    {
        YAML::Node meta;
        if (!m_baseIO->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
        {
            std::cout << "Failed to load meta data" << std::endl;
            return ret;
        }
        ret = std::make_shared<Camera>(meta.as<Camera>());
    }
    else
    {
        ret = std::make_shared<Camera>();
    }

    /// DATA
    size_t scanGroupNo = 0;
    while(true)
    {
        CameraImageGroupPtr group = m_cameraImageGroupIO->load(scanPosNo, scanCamNo, scanGroupNo);
        if (group)
        {
            ret->groups.push_back(group);
        }
        else if(scanGroupNo > 1)
        {
            break;
        }
        scanGroupNo++;
    }
    return ret;
}

template <typename BaseIO>
boost::optional<YAML::Node> CameraIO<BaseIO>::loadMeta(
    const size_t& scanPosNo,
    const size_t& scanCamNo) const
{
    Description d = m_baseIO->m_description->camera(scanPosNo, scanCamNo);
    return m_metaIO->load(d);
}

template <typename BaseIO>
void CameraIO<BaseIO>::saveCamera(
    const size_t& scanPosNo, 
    const size_t& scanCamNo, 
    CameraPtr cameraPtr) const
{
    save(scanPosNo, scanCamNo, cameraPtr);
}

template <typename BaseIO>
CameraPtr CameraIO<BaseIO>::loadCamera(
    const size_t& scanPosNo, const size_t& scanCamNo) const
{
    return load(scanPosNo, scanCamNo);
}

} // namespace scanio

} // namespace lvr2
