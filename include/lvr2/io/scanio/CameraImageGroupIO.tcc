namespace lvr2
{

namespace scanio
{

template <typename BaseIO>
void CameraImageGroupIO<BaseIO>::save(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& groupNo,
    CameraImageGroupPtr imgPtr) const
{
    // Get schema 
    auto Dgen = m_baseIO->m_description;
    Description d = Dgen->cameraImageGroup(scanPosNo, camNo, groupNo);

    // Save image data
    for (size_t i = 0; i < imgPtr->images.size(); i++)
    {
        CameraImagePtr img = imgPtr->images[i];
        m_cameraImageIO->save(scanPosNo, camNo, groupNo, i, img);
    }

    // Save meta data
    if (d.meta)
    {
        YAML::Node node;
        node = *imgPtr;
        m_baseIO->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
    }
}

template <typename BaseIO>
void CameraImageGroupIO<BaseIO>::save(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& groupNo,
    const std::vector<size_t>& imgNos,
    CameraImageGroupPtr imgPtr) const
{
    auto Dgen = m_baseIO->m_description;
    Description d = Dgen->cameraImageGroup(scanPosNo, camNo, groupNo);

    // Save only given image ids
    for(auto i : imgNos)
    {
        if(i < imgPtr->images.size())
        {
            CameraImagePtr img = imgPtr->images[i];
            m_cameraImageIO->save(scanPosNo, camNo, groupNo, img);
        }
    }
}

template <typename BaseIO>
CameraImageGroupPtr CameraImageGroupIO<BaseIO>::load(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& groupNo) const
{
    CameraImageGroupPtr ret;

    auto Dgen = m_baseIO->m_description;
    Description d = Dgen->cameraImageGroup(scanPosNo, camNo, groupNo);

    if (!d.dataRoot)
    {
        return ret;
    }

    if (!m_baseIO->m_kernel->exists(*d.dataRoot))
    {
        return ret;
    }

    if (d.meta)
    {
        YAML::Node meta;
        if (!m_baseIO->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
        {
            return ret;
        }

        CameraImageGroupPtr group(new CameraImageGroup);

        if (YAML::convert<CameraImageGroup>::decode(meta, *group))
        {
            // success
            ret = group;
        }
        else
        {
            // meta seems to be some other type: return empty pointer
            return ret;
        }
    }
    else
    {
        ret = std::make_shared<CameraImageGroup>();
    }

    if (ret)
    {
        // it is a group!
        // load data
        for (size_t i = 0;; i++)
        {
            CameraImagePtr img = m_cameraImageIO->load(scanPosNo, camNo, groupNo, i);
            
            if (img)
            {
                ret->images.push_back(img);
            }
            // Always test image 0 and 1. If numbering
            // starts with i >= 2, data will not be found
            else if(i > 1)
            {
                break;
            }
        }
    }
    return ret;
}

template <typename BaseIO>
boost::optional<YAML::Node> CameraImageGroupIO<BaseIO>::loadMeta(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& groupNo) const
{
     Description d = m_baseIO->m_description->cameraImageGroup(scanPosNo, camNo, groupNo);
     return m_metaIO->load(d);
}


// template <typename BaseIO>
// void CameraImageGroupIO<BaseIO>::saveCameraImage(
//     const size_t& scanPosNr, 
//     const size_t& camNr, 
//     const size_t& imgNr, 
//     CameraImageGroupPtr imgPtr) const
// {
//     save(scanPosNr, camNr, imgNr, imgPtr);
// }

// template <typename BaseIO>
// CameraImageGroupPtr CameraImageGroupIO<BaseIO>::loadCameraImage(
//     const size_t& scanPosNr, 
//     const size_t& camNr, 
//     const size_t& imgNr) const
// {
//     return load(scanPosNr, camNr, imgNr);
// }

} // namespace scanio

} // namespace lvr2
