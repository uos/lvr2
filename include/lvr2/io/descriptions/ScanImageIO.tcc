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
    
    //TODO load data

    // loading
    // should data be loaded ?
    boost::optional<cv::Mat> opt_img = m_imageIO->loadImage(*d.groupName, *d.dataSetName);
    if(opt_img)
    {
        ret->image = *opt_img;
    }

    // std::cout << "Loading Scan Image "<< std::endl;
    // std::cout << d << std::endl;

    // ret->imageFile = *d.dataSetName;

    return ret;
}

std::string shortestPath(std::string from, std::string to)
{
    boost::filesystem::path root(".");
    boost::filesystem::path to_path(to);
    boost::filesystem::path from_path(from);
    boost::filesystem::path ret = (root / to_path).lexically_relative(root / from_path.parent_path());
    return ret.string();
}

template <typename FeatureBase>
void  ScanImageIO<FeatureBase>::saveScanImage(
    const size_t& scanPos, 
    const size_t& camNr, 
    const size_t& imgNr, 
    ScanImagePtr imgPtr) const
{
    Description d = m_featureBase->m_description->scanImage(scanPos, camNr, imgNr);

    if(d.metaName)
    {
        // add image file path relative to meta: 
        // TODO: find shortest path from meta to dataset
        // imgPtr->imageFile = *d.dataSetName;
        // imgPtr->imageFile = shortestPath(*d.metaName, *d.dataSetName);
        YAML::Node node;
        node = *imgPtr;
        m_featureBase->m_kernel->saveMetaYAML(*d.groupName, *d.metaName, node);
    }

    // std::cout << "Saving Scan Image "<< std::endl;
    // std::cout << d << std::endl;

    m_imageIO->save(*d.groupName, *d.dataSetName, imgPtr->image);
}

} // namespace lvr2
