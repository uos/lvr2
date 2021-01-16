#include "lvr2/io/yaml/ScanImage.hpp"
namespace lvr2
{


template <typename FeatureBase>
ScanImagePtr ScanImageIO<FeatureBase>::loadScanImage(
    const size_t& scanPos, 
    const size_t& camNr, 
    const size_t& imgNr)
{
    ScanImagePtr ret;

    Description d = m_featureBase->m_description->scanImage(scanPos, camNr, imgNr);

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
        ret = std::make_shared<ScanImage>(meta.as<ScanImage>());
    } else {
        ret.reset(new ScanImage);
    }

    ret->imageFile = *d.dataSetName;
    
    //TODO load data

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
void  ScanImageIO<FeatureBase>::saveScanImage(
    const size_t& scanPos, 
    const size_t& camNr, 
    const size_t& imgNr, 
    ScanImagePtr imgPtr) const
{
    // TODO
    Description d = m_featureBase->m_description->scanImage(scanPos, camNr, imgNr);

    // std::cout << "[ScanImageIO] Image " << scanPos << "," << camNr << "," << imgNr <<  " - Description: " << std::endl;
    // std::cout << d << std::endl;

    if(d.metaName)
    {
        // add image file to meta
        imgPtr->imageFile = *d.dataSetName;
        YAML::Node node;
        node = *imgPtr;
        m_featureBase->m_kernel->saveMetaYAML(*d.groupName, *d.metaName, node);
    }

    m_imageIO->save(*d.groupName, *d.dataSetName, imgPtr->image);
}

} // namespace lvr2
