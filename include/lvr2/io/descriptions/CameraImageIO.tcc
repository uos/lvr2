#include "lvr2/io/yaml/CameraImage.hpp"
namespace lvr2
{

template <typename FeatureBase>
void CameraImageIO<FeatureBase>::save(
    const size_t& scanPosNr,
    const size_t& camNr,
    const size_t& imgNr,
    CameraImagePtr imgPtr) const
{
    auto D = m_featureBase->m_description;
    Description d = D->cameraImage(
        D->camera( D->position(scanPosNr), camNr), imgNr);

    m_imageIO->save(*d.groupName, *d.dataSetName, imgPtr->image);

    if(d.metaName)
    {
        YAML::Node node;
        node = *imgPtr;
        m_featureBase->m_kernel->saveMetaYAML(*d.groupName, *d.metaName, node);
    }
}

template <typename FeatureBase>
CameraImagePtr CameraImageIO<FeatureBase>::load(
    const size_t& scanPosNr,
    const size_t& camNr,
    const size_t& imgNr) const
{
    CameraImagePtr ret;

    auto Dgen = m_featureBase->m_description;
    Description d_pos = Dgen->position(scanPosNr);
    Description d_cam = Dgen->camera(d_pos, camNr);
    Description d = Dgen->cameraImage(d_cam, imgNr);

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
        if(!m_featureBase->m_kernel->exists(*d.groupName, *d.metaName))
        {
            return ret;
        }
        
        YAML::Node meta;
        m_featureBase->m_kernel->loadMetaYAML(*d.groupName, *d.metaName, meta);
        ret = std::make_shared<CameraImage>(meta.as<CameraImage>());
    } else {
        ret.reset(new CameraImage);
    }
    
    // loading
    // should data be loaded ?
    boost::optional<cv::Mat> opt_img = m_imageIO->loadImage(*d.groupName, *d.dataSetName);
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
