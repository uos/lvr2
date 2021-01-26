#include "lvr2/io/yaml/ScanPosition.hpp"

 #include <boost/optional/optional_io.hpp>

namespace lvr2
{

template <typename  FeatureBase>
void ScanPositionIO< FeatureBase>::save(
    const size_t& scanPosNo, 
    ScanPositionPtr scanPositionPtr) const
{
    Description d = m_featureBase->m_description->position(scanPosNo);
  
    // std::cout << "[ScanPositionIO] ScanPosition " << scanPosNo << " - Description: " << std::endl;
    // std::cout << d << std::endl;

    // Save meta information
    if(d.metaName)
    {
        YAML::Node node;
        node = *scanPositionPtr;
        m_featureBase->m_kernel->saveMetaYAML(*d.groupName, *d.metaName, node);
    }
    
    // Save all lidar sensors
    for(size_t i = 0; i < scanPositionPtr->lidars.size(); i++)
    {
        m_lidarIO->save(scanPosNo, i, scanPositionPtr->lidars[i]);
    }

    // Save all scan camera sensors
    for(size_t i = 0; i < scanPositionPtr->cameras.size(); i++)
    {
        std::cout << " [ScanPositionIO]: Writing camera " << i << std::endl;
        m_cameraIO->saveCamera(scanPosNo, i, scanPositionPtr->cameras[i]);
    }
    
    // Save hyperspectral data
    // if (scanPositionPtr->hyperspectralCamera)
    // {
    //     m_hyperspectralCameraIO->saveHyperspectralCamera(scanPosNo, scanPositionPtr->hyperspectralCamera);
    // }
}

template <typename  FeatureBase>
void ScanPositionIO< FeatureBase>::saveScanPosition(
    const size_t& scanPosNo, 
    ScanPositionPtr scanPositionPtr) const
{
    save(scanPosNo, scanPositionPtr);
}

template <typename  FeatureBase>
ScanPositionPtr ScanPositionIO< FeatureBase>::loadScanPosition(
    const size_t& scanPosNo) const
{
    ScanPositionPtr ret;

    Description d = m_featureBase->m_description->position(scanPosNo);

    if(!m_featureBase->m_kernel->exists(*d.groupName))
    {
        return ret;
    }

    // std::cout << "[ScanPositionIO - load] Description:" << std::endl;
    // std::cout << d << std::endl;

    // Setup defaults
    if(d.metaName)
    {
        if(!m_featureBase->m_kernel->exists(*d.groupName, *d.metaName))
        {
            std::cout << timestamp << " [ScanPositionIO]: Specified meta file not found. " << std::endl;
            return ret;
        } 

        YAML::Node meta;
        m_featureBase->m_kernel->loadMetaYAML(*d.groupName, *d.metaName, meta);
        ret = std::make_shared<ScanPosition>(meta.as<ScanPosition>());
        
    } else {
        // no meta name specified but scan position is there: 
        ret.reset(new ScanPosition);
    }
    
    // Get all lidar sensors
    size_t lidarNo = 0;
    while(true)
    {
        LIDARPtr lidar = m_lidarIO->load(scanPosNo, lidarNo);
        
        if(lidar)
        {
            ret->lidars.push_back(lidar);
        } else {
            break;
        }

        // std::cout << "[ScanPositionIO - load] Loaded Scan " << scanNo << std::endl;

        ++lidarNo;
    }

    // TODO: make below lines same as above ones
    // let the features decide if data is available

    // Get all scan cameras
    size_t camNo = 0;
    while(true)
    {
        CameraPtr cam = m_cameraIO->loadCamera(scanPosNo, camNo);
        if(cam)
        {
            ret->cameras.push_back(cam);
        } else {
            break;
        }

        // std::cout << "[ScanPositionIO - load] Loaded Camera " << camNo << std::endl;

        camNo++;
    }

    // Get hyperspectral data
    // Description hyperDescr = m_featureBase->m_description->hyperspectralCamera(scanPosNo);
    // if(hyperDescr.dataSetName)
    // {
    //     std::string groupName;
    //     std::string dataSetName;
    //     std::tie(groupName, dataSetName) = getNames("", "", hyperDescr);

    //     if (m_featureBase->m_kernel->exists(groupName))
    //     {
    //         std::cout << timestamp << "ScanPositionIO: Loading hyperspectral data... " << std::endl;
    //         HyperspectralCameraPtr hspCam = m_hyperspectralCameraIO->loadHyperspectralCamera(scanPosNo);
    //         ret->hyperspectralCamera = hspCam;
    //     }
    // }

    return ret;
}

template <typename  FeatureBase>
ScanPositionPtr ScanPositionIO< FeatureBase>::loadScanPosition(
    const size_t& scanPosNo, ReductionAlgorithmPtr reduction) const
{
    // TODO
}

} // namespace lvr2
