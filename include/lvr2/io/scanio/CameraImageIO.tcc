namespace lvr2
{

namespace scanio
{

template <typename FeatureBase>
void CameraImageIO<FeatureBase>::save(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& imgNo,
    CameraImagePtr imgPtr) const
{
    std::vector<size_t> imgNos = {imgNo};
    save(scanPosNo, camNo, imgNos, imgPtr);
}

template <typename FeatureBase>
void CameraImageIO<FeatureBase>::save(
    const size_t& scanPosNo,
    const size_t& camNo,
    const std::vector<size_t>& imgNos,
    CameraImagePtr imgPtr) const
{
    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->cameraImage(scanPosNo, camNo, imgNos);

    // std::cout << "[CameraImageIO - save] Description:" << std::endl;
    // std::cout << d << std::endl;


    // save data

    const bool data_loaded_before = imgPtr->loaded();

    if(!data_loaded_before)
    {
        imgPtr->load();
    }

    m_imageIO->save(*d.dataRoot, *d.data, imgPtr->image);

    if(!data_loaded_before)
    {
        imgPtr->release();
    }

    // save meta
    if(d.meta)
    {
        YAML::Node node;
        node = *imgPtr;
        m_featureBase->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
    }
}

template <typename FeatureBase>
CameraImagePtr CameraImageIO<FeatureBase>::load(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& imgNo) const
{
    std::vector<size_t> imgNos = {imgNo};
    return load(scanPosNo, camNo, imgNos);
}

template <typename FeatureBase>
CameraImagePtr CameraImageIO<FeatureBase>::load(
    const size_t& scanPosNo,
    const size_t& camNo,
    const std::vector<size_t>& imgNos) const
{
    CameraImagePtr ret;

    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->cameraImage(scanPosNo, camNo, imgNos);

    if(!d.dataRoot)
    {
        return ret;
    }

    if(!m_featureBase->m_kernel->exists(*d.dataRoot))
    {
        return ret;
    }
    
    if(d.meta)
    {   
        YAML::Node meta;
        if(!m_featureBase->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
        {
            return ret;
        }

        CameraImagePtr loaded(new CameraImage);
        if(YAML::convert<CameraImage>::decode(meta, *loaded) )
        {
            // success
            ret = loaded;
        } else {
            // cannot decode
            return ret;
        }
        
    } else {
        ret.reset(new CameraImage);
    }

    // should we load the data here?
    // we could pass the loader as link to the storage instead of the data now
    std::function<cv::Mat()> image_loader = [
        schema = m_featureBase->m_description,
        kernel = m_featureBase->m_kernel,
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

    if(m_featureBase->m_load_data)
    {
        ret->image = image_loader();
    }

    return ret;
}

template <typename FeatureBase>
boost::optional<YAML::Node> CameraImageIO<FeatureBase>::loadMeta(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& imgNo) const
{
    std::vector<size_t> imgNos = {imgNo};
    return loadMeta(scanPosNo, camNo, imgNos);
}

template <typename FeatureBase>
boost::optional<YAML::Node> CameraImageIO<FeatureBase>::loadMeta(
    const size_t& scanPosNo,
    const size_t& camNo,
    const std::vector<size_t>& imgNos) const
{
    Description d = m_featureBase->m_description->cameraImage(scanPosNo, camNo, imgNos);
    return m_metaIO->load(d); 
}

template <typename FeatureBase>
void CameraImageIO<FeatureBase>::saveCameraImage(
    const size_t& scanPosNr, 
    const size_t& camNr, 
    const size_t& imgNr, 
    CameraImagePtr imgPtr) const
{
    save(scanPosNr, camNr, imgNr, imgPtr);
}

template <typename FeatureBase>
CameraImagePtr CameraImageIO<FeatureBase>::loadCameraImage(
    const size_t& scanPosNr, 
    const size_t& camNr, 
    const size_t& imgNr) const
{
    return load(scanPosNr, camNr, imgNr);
}

} // namespace scanio

} // namespace lvr2
