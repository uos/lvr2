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
    ScanCameraPtr& camera)
{
    // TODO
}

template <typename FeatureBase>
ScanCameraPtr ScanCameraIO<FeatureBase>::loadScanCamera(
    const size_t& scanPosNo, const size_t& scanCamNo)
{
    ScanCameraPtr ret(new ScanCamera);

    Description d = m_featureBase->m_description->scanCamera(scanPosNo, scanCamNo);

    if(d.metaData)
    {
        *ret = (*d.metaData).as<ScanCamera>();
    }
    else
    {
        std::cout << timestamp << "ScanCameraIO::loadScanCamera(): Warning: No meta data found for cam_"
                  << scanCamNo << "." << std::endl;
    }

    std::string groupName;
    std::string dataSetName;

    size_t scanImageNo = 0;
    do
    {
        Description scanImageDescr = m_featureBase->m_description->scanImage(scanPosNo, 0, scanCamNo, scanImageNo);
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
