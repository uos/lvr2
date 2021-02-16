#include "lvr2/io/yaml/CameraImage.hpp"
namespace lvr2
{

template <typename FeatureBase>
void CameraImageIO<FeatureBase>::save(
    const size_t& scanPosNo,
    const size_t& camNo,
    const size_t& imgNo,
    CameraImagePtr imgPtr) const
{
    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->cameraImage(scanPosNo, camNo, imgNo);

    // std::cout << "[CameraImageIO - save] Description:" << std::endl;
    // std::cout << d << std::endl;


    // save data
    m_imageIO->save(*d.dataRoot, *d.data, imgPtr->image);

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
    CameraImagePtr ret;

    auto Dgen = m_featureBase->m_description;
    Description d = Dgen->cameraImage(scanPosNo, camNo, imgNo);

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
        ret = std::make_shared<CameraImage>(meta.as<CameraImage>());
    } else {
        ret.reset(new CameraImage);
    }
    
    // loading
    // should data be loaded ?
    boost::optional<cv::Mat> opt_img = m_imageIO->loadImage(*d.dataRoot, *d.data);
    if(opt_img)
    {
        ret->image = *opt_img;
    }

    return ret;
}

template <typename FeatureBase>
void CameraImageIO<FeatureBase>::saveCameraImage(
    const size_t& scanPosNr, 
    const size_t& camNr, 
    const size_t& imgNr, 
    CameraImagePtr imgPtr) const
{
    save(scanPosNr, camNr, imgNr);
}

template <typename FeatureBase>
CameraImagePtr CameraImageIO<FeatureBase>::loadCameraImage(
    const size_t& scanPosNr, 
    const size_t& camNr, 
    const size_t& imgNr) const
{
    return load(scanPosNr, camNr, imgNr);
}

} // namespace lvr2
