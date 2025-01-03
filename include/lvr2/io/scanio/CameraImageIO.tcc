namespace lvr2
{

namespace scanio
{

template <typename BaseIO>
void CameraImageIO<BaseIO>::save(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& groupNo,
    const size_t& imgNo,
    CameraImagePtr imgPtr) const
{
    // std::vector<size_t> imgNos = {imgNo};
    // save(scanPosNo, camNo, groupNo, imgNos, imgPtr);

    auto Dgen = m_baseIO->m_description;

    Description d = Dgen->cameraImage(scanPosNo, camNo, groupNo, imgNo);

    const bool data_loaded_before = imgPtr->loaded();

    if (!data_loaded_before)
    {
        imgPtr->load();
    }

    m_imageIO->save(*d.dataRoot, *d.data, imgPtr->image);

    if (!data_loaded_before)
    {
        imgPtr->release();
    }

    // save meta
    if (d.meta)
    {
        YAML::Node node;
        node = *imgPtr;
        m_baseIO->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
    }
}

// template <typename BaseIO>
// void CameraImageIO<BaseIO>::save(
//     const size_t& scanPosNo,
//     const size_t& camNo,
//     const size_t& groupNo,
//     const std::vector<size_t>& imgNos,
//     const std::vector<CameraImagePtr>& imgPtr) const
// {
//     auto Dgen = m_baseIO->m_description;

//     if (imgNos.size() == imgPtr.size())
//     {
//         for(auto i : imgNos)
//         {
//             Description d = Dgen->cameraImage(scanPosNo, camNo, groupNo, i);

//             // std::cout << "[CameraImageIO - save] Description:" << std::endl;
//             // std::cout << d << std::endl;

//             // save data

//             const bool data_loaded_before = imgPtr[i]->loaded();

//             if (!data_loaded_before)
//             {
//                 imgPtr[i]->load();
//             }

//             m_imageIO->save(*d.dataRoot, *d.data, imgPtr[i]->image);

//             if (!data_loaded_before)
//             {
//                 imgPtr[i]->release();
//             }

//             // save meta
//             if (d.meta)
//             {
//                 YAML::Node node;
//                 node = *imgPtr[i];
//                 m_baseIO->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
//             }
//         }
//     }
// }

template <typename BaseIO>
CameraImagePtr CameraImageIO<BaseIO>::load(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& groupNo,
    const size_t& imgNo) const
{
    CameraImagePtr ret;

    auto Dgen = m_baseIO->m_description;
    Description d = Dgen->cameraImage(scanPosNo, camNo, groupNo, imgNo);

    if(!d.dataRoot)
    {
        return ret;
    }

    if(!m_baseIO->m_kernel->exists(*d.dataRoot))
    {
        return ret;
    }

    if(!d.data)
    {
        return ret;
    }

    if(!m_baseIO->m_kernel->exists(*d.dataRoot, *d.data))
    {
        return ret;
    }
    
    if(d.meta)
    {   
        YAML::Node meta;
        if(!m_baseIO->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
        {
            return ret;
        }

        CameraImagePtr loaded = std::make_shared<CameraImage>();
        if (YAML::convert<CameraImage>::decode(meta, *loaded))
        {
            // success
            ret = loaded;
        }
        else
        {
            // cannot decode
            return ret;
        }
    }
    else
    {
        ret = std::make_shared<CameraImage>();
    }

    // should we load the data here?
    // we could pass the loader as link to the storage instead of the data now
    std::function<cv::Mat()> image_loader = [
        schema = m_baseIO->m_description,
        kernel = m_baseIO->m_kernel,
        d]()
    {
        FeatureBuild<CameraImageIO> io(kernel, schema, false);

        cv::Mat ret;
        boost::optional<cv::Mat> opt_img = io.ImageIO::load(*d.dataRoot, *d.data);
        if(opt_img)
        {
            ret = *opt_img;
        }
        return ret;
    };

    // TODO: add this function to the struct
    // Old:
    // ret->image = image_loader();
    // New:
    ret->image_loader = image_loader;

    if(m_baseIO->m_load_data)
    {
        ret->image = image_loader();
    }

    return ret;
}

template <typename BaseIO>
std::vector<CameraImagePtr> CameraImageIO<BaseIO>::load(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& groupNo,
    const std::vector<size_t>& imgNos) const
{
    std::vector<CameraImagePtr> imgs;

    for(auto i : imgNos)
    {
        CameraImagePtr img = load(scanPosNo, camNo, groupNo, i);
        if(img)
        {
            imgs.push_back(img);
        }
    }

    return imgs;
}

template <typename BaseIO>
boost::optional<YAML::Node> CameraImageIO<BaseIO>::loadMeta(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& groupNo,
    const size_t& imgNo) const
{
    Description d = m_baseIO->m_description->cameraImage(scanPosNo, camNo, groupNo, imgNo);
    return m_metaIO->load(d); 
}

template <typename BaseIO>
boost::optional<YAML::Node> CameraImageIO<BaseIO>::loadMeta(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& groupNo,
    const std::vector<size_t>& imgNos) const
{
    // TODO: Re-think if this function is really neccessary...
    Description d = m_baseIO->m_description->cameraImage(scanPosNo, camNo, groupNo, imgNos);
    return m_metaIO->load(d); 
}

template <typename BaseIO>
void CameraImageIO<BaseIO>::saveCameraImage(
    const size_t& scanPosNr, 
    const size_t& camNr, 
    const size_t& groupNo,
    const size_t& imgNr, 
    CameraImagePtr imgPtr) const
{
    save(scanPosNr, camNr, groupNo, imgNr, imgPtr);
}

template <typename BaseIO>
CameraImagePtr CameraImageIO<BaseIO>::loadCameraImage(
    const size_t& scanPosNr, 
    const size_t& camNr, 
    const size_t& groupNo,
    const size_t& imgNr) const
{
    return load(scanPosNr, camNr, groupNo, imgNr);
}

} // namespace scanio

} // namespace lvr2
