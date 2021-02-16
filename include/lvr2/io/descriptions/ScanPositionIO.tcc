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

    // Save all lidar sensors
    for(size_t i = 0; i < scanPositionPtr->lidars.size(); i++)
    {
        m_lidarIO->save(scanPosNo, i, scanPositionPtr->lidars[i]);
    }

    // std::cout << "[ScanPositionIO] LIDARs written. " << std::endl;

    // Save all scan camera sensors
    for(size_t i = 0; i < scanPositionPtr->cameras.size(); i++)
    {
        // std::cout << " [ScanPositionIO]: Writing camera " << i << std::endl;
        m_cameraIO->save(scanPosNo, i, scanPositionPtr->cameras[i]);
    }

    // std::cout << "[ScanPositionIO] Cameras written. " << std::endl;
    
    // Save all hyperspectral camera sensors
    for(size_t i=0; i < scanPositionPtr->hyperspectral_cameras.size(); i++)
    {
        m_hyperspectralCameraIO->save(scanPosNo, i, scanPositionPtr->hyperspectral_cameras[i]);
    }

    // std::cout << "[ScanPositionIO - save] Write Meta." << std::endl;
    // Save meta information
    if(d.meta)
    {
        YAML::Node node;
        node = *scanPositionPtr;
        m_featureBase->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
    }

    // std::cout << "[ScanPositionIO] Meta written. " << std::endl;
}


template <typename  FeatureBase>
ScanPositionPtr ScanPositionIO< FeatureBase>::load(
    const size_t& scanPosNo) const
{
    ScanPositionPtr ret;

    Description d = m_featureBase->m_description->position(scanPosNo);

    // std::cout << "[ScanPositionIO - load]"  << std::endl;
    // std::cout << d <<  std::endl;

    // Check if specified scan position exists
    if(!m_featureBase->m_kernel->exists(*d.dataRoot))
    {
        return ret;
    }

    if(d.meta)
    {
        if(!m_featureBase->m_kernel->exists(*d.metaRoot, *d.meta))
        {
            std::cout << timestamp << " [ScanPositionIO]: Specified meta file not found. " << std::endl;
            return ret;
        } 

        YAML::Node meta;
        m_featureBase->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta);
        ret = std::make_shared<ScanPosition>(meta.as<ScanPosition>());
    } else {
        // no meta name specified but scan position is there: 
        ret.reset(new ScanPosition);
    }

    //// DATA

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

    // Get all scan cameras
    size_t camNo = 0;
    while(true)
    {
        CameraPtr cam = m_cameraIO->load(scanPosNo, camNo);
        if(cam)
        {
            ret->cameras.push_back(cam);
        } else {
            break;
        }

        // std::cout << "[ScanPositionIO - load] Loaded Camera " << camNo << std::endl;

        camNo++;
    }

    // Get all hyperspectral cameras
    size_t hCamNo = 0;
    while(true)
    {
        HyperspectralCameraPtr cam = m_hyperspectralCameraIO->load(scanPosNo, hCamNo);
        if(cam)
        {
            ret->hyperspectral_cameras.push_back(cam);
        } else {
            break;
        }

        hCamNo++;
    }

    return ret;
}

template <typename  FeatureBase>
ScanPositionPtr ScanPositionIO< FeatureBase>::load(
    const size_t& scanPosNo, ReductionAlgorithmPtr reduction) const
{
    ScanPositionPtr ret = load(scanPosNo);

    if(ret)
    {
        if(reduction)
        {
            // TODO
        }
    }

    return ret;
}

template <typename  FeatureBase>
void ScanPositionIO< FeatureBase>::saveScanPosition(
    const size_t& scanPosNo, 
    ScanPositionPtr scanPositionPtr) const
{
    save(scanPosNo, scanPositionPtr);
}

template <typename  FeatureBase>
ScanPositionPtr ScanPositionIO<FeatureBase>::loadScanPosition(
    const size_t& scanPosNo) const
{
    return load(scanPosNo);
}

template <typename  FeatureBase>
ScanPositionPtr ScanPositionIO< FeatureBase>::loadScanPosition(
    const size_t& scanPosNo, ReductionAlgorithmPtr reduction) const
{
    return load(scanPosNo, reduction);
}

} // namespace lvr2
