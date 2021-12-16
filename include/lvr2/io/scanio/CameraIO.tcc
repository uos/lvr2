namespace lvr2
{

template <typename FeatureBase>
void CameraIO<FeatureBase>::save(
    const size_t& scanPosNo,
    const size_t& scanCamNo,
    CameraPtr cameraPtr) const
{
    auto Dgen = m_featureBase->m_description;

    Description d = Dgen->camera(scanPosNo, scanCamNo);

    if(!d.dataRoot)
    {
        // someone doesnt want to save cameras
        return;
    }

    // writing data
    for(size_t scanImageNo = 0; scanImageNo < cameraPtr->images.size(); scanImageNo++)
    {
        if(cameraPtr->images[scanImageNo].is_type<CameraImagePtr>())
        {
            // Image
            CameraImagePtr img;
            img <<= cameraPtr->images[scanImageNo];
            m_cameraImageIO->save(scanPosNo, scanCamNo, scanImageNo, img);
        } else {
            // Group
            CameraImageGroupPtr group;
            group <<= cameraPtr->images[scanImageNo];
            m_cameraImageGroupIO->save(scanPosNo, scanCamNo, scanImageNo, group);
        }
    }

    // writing meta
    if(d.meta)
    {
        YAML::Node node;
        node = *cameraPtr;
        m_featureBase->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
    }
}

template <typename FeatureBase>
CameraPtr CameraIO<FeatureBase>::load(
    const size_t& scanPosNo, 
    const size_t& scanCamNo) const
{
    CameraPtr ret;

    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->camera(scanPosNo, scanCamNo);

    if(!d.dataRoot)
    {
        return ret;
    }

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
            std::cout << meta << std::endl;
            return ret;
        }
        ret = std::make_shared<Camera>(meta.as<Camera>());
    } else {
        ret.reset(new Camera);
    }

    /// DATA
    size_t scanImageNo = 0;
    while(true)
    {
        CameraImagePtr image = m_cameraImageIO->load(scanPosNo, scanCamNo, scanImageNo);
        
        if(image)
        {
            ret->images.push_back(image);
        } else {
            CameraImageGroupPtr group = m_cameraImageGroupIO->load(scanPosNo, scanCamNo, scanImageNo);

            if(group)
            {
                ret->images.push_back(group);
            } else {
                break;
            }
        }
        scanImageNo++;
    }

    return ret;
}

template <typename FeatureBase>
boost::optional<YAML::Node> CameraIO<FeatureBase>::loadMeta(
    const size_t& scanPosNo,
    const size_t& scanCamNo) const
{
    Description d = m_featureBase->m_description->camera(scanPosNo, scanCamNo);
    return m_metaIO->load(d);
}

template <typename FeatureBase>
void CameraIO<FeatureBase>::saveCamera(
    const size_t& scanPosNo, 
    const size_t& scanCamNo, 
    CameraPtr cameraPtr) const
{
    save(scanPosNo, scanCamNo, cameraPtr);
}

template <typename FeatureBase>
CameraPtr CameraIO<FeatureBase>::loadCamera(
    const size_t& scanPosNo, const size_t& scanCamNo) const
{
    return load(scanPosNo, scanCamNo);
}

} // namespace lvr2
