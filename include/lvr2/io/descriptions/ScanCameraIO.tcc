#include "lvr2/io/yaml/ScanCamera.hpp"
namespace lvr2
{

// template <typename FeatureBase>
// void ScanCameraIO<FeatureBase>::save(
//     const std::string& group, 
//     const std::string& container, ScanCameraPtr& buffer)
// {
//     // TODO
// }

template <typename FeatureBase>
void ScanCameraIO<FeatureBase>::saveScanCamera(
    const size_t& scanPosNo, const size_t& scanCamNo, 
    ScanCameraPtr cameraPtr) const
{
    // TODO
    Description d = m_featureBase->m_description->scanCamera(scanPosNo, scanCamNo);

    std::cout << "[ScanCameraIO] Cam " << scanPosNo << "," << scanCamNo <<  " - Description: " << std::endl;
    std::cout << d << std::endl;

    if(d.metaName)
    {
        YAML::Node node;
        node = *cameraPtr;
        m_featureBase->m_kernel->saveMetaYAML(*d.groupName, *d.metaName, node);
    }


    for(size_t scanImageNo = 0; scanImageNo < cameraPtr->images.size(); scanImageNo++)
    {
        std::cout << "Saving image " << scanImageNo << std::endl;
        m_scanImageIO->saveScanImage(scanPosNo, scanCamNo, scanImageNo, cameraPtr->images[scanImageNo]);
    }
    
}

template <typename FeatureBase>
ScanCameraPtr ScanCameraIO<FeatureBase>::loadScanCamera(
    const size_t& scanPosNo, const size_t& scanCamNo)
{
    ScanCameraPtr ret;

    Description d = m_featureBase->m_description->scanCamera(scanPosNo, scanCamNo);

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
        ret = std::make_shared<ScanCamera>(meta.as<ScanCamera>());
    } else {
        ret.reset(new ScanCamera);
    }

    std::string groupName;
    std::string dataSetName;

    size_t scanImageNo = 0;
    do
    {
        Description scanImageDescr = m_featureBase->m_description->scanImage(scanPosNo, scanCamNo, scanImageNo);
        std::tie(groupName, dataSetName) = getNames("", "", scanImageDescr);
        if(m_featureBase->m_kernel->exists(groupName, dataSetName))
        {
            ScanImagePtr scanImage = m_scanImageIO->loadScanImage(scanPosNo, scanCamNo, scanImageNo);
            ret->images.push_back(scanImage);
        }
        else
        {
            break;
        }
        scanImageNo++;
    } while (true);

    return ret;
}



template <typename FeatureBase>
bool ScanCameraIO<FeatureBase>::isScanCamera(const std::string& group)
{
    return true;
}

} // namespace lvr2
