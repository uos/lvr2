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
    
    // Save all scans
    for(size_t i = 0; i < scanPositionPtr->scans.size(); i++)
    {
        m_scanIO->saveScan(scanPosNo, i, scanPositionPtr->scans[i]);
    }

    // Save all scan camera and images
    for(size_t i = 0; i < scanPositionPtr->cams.size(); i++)
    {
        m_scanCameraIO->saveScanCamera(scanPosNo, i, scanPositionPtr->cams[i]);
    }
    
    // Save hyperspectral data
    if (scanPositionPtr->hyperspectralCamera)
    {
        m_hyperspectralCameraIO->saveHyperspectralCamera(scanPosNo, scanPositionPtr->hyperspectralCamera);
    }
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

    // std::cout << "[ScanPositionIO] load() with Description:" << std::endl;
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
    
    // Get all sub scans
    size_t scanNo = 0;
    while(true)
    {
        ScanPtr scan = m_scanIO->loadScan(scanPosNo, scanNo);
        
        if(scan)
        {
            ret->scans.push_back(scan);
        } else {
            break;
        }

        ++scanNo;
    }

    // TODO: make below lines same as above ones
    // let the features decide if data is available

    // Get all scan cameras
    size_t camNo = 0;
    while(true)
    {
        ScanCameraPtr cam = m_scanCameraIO->loadScanCamera(scanPosNo, camNo);
        if(cam)
        {
            ret->cams.push_back(cam);
        } else {
            break;
        }
        camNo++;
    }

    // Get hyperspectral data
    Description hyperDescr = m_featureBase->m_description->hyperspectralCamera(scanPosNo);
    if(hyperDescr.dataSetName)
    {
        std::string groupName;
        std::string dataSetName;
        std::tie(groupName, dataSetName) = getNames("", "", hyperDescr);

        if (m_featureBase->m_kernel->exists(groupName))
        {
            std::cout << timestamp << "ScanPositionIO: Loading hyperspectral data... " << std::endl;
            HyperspectralCameraPtr hspCam = m_hyperspectralCameraIO->loadHyperspectralCamera(scanPosNo);
            ret->hyperspectralCamera = hspCam;
        }
    }

    return ret;
}

template <typename  FeatureBase>
ScanPositionPtr ScanPositionIO< FeatureBase>::loadScanPosition(
    const size_t& scanPosNo, ReductionAlgorithmPtr reduction) const
{
    ScanPositionPtr ret;

    Description d = m_featureBase->m_description->position(scanPosNo);

    if(!m_featureBase->m_kernel->exists(*d.groupName))
    {
        return ret;
    }

    std::cout << "[ScanPositionIO] load() with Description:" << std::endl;
    std::cout << d << std::endl;

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
    
    // Get all sub scans
    size_t scanNo = 0;
    while(true)
    {
        ScanPtr scan = m_scanIO->loadScan(scanPosNo, scanNo, reduction);
        
        if(scan)
        {
            ret->scans.push_back(scan);
        } else {
            break;
        }

        ++scanNo;
    }


    // TODO: make below lines same as above ones
    // let the features decide if data is available

    // Get all scan cameras
    size_t camNo = 0;
    do
    {
        // Get description for next scan
        Description camDescr = m_featureBase->m_description->scanCamera(scanPosNo, camNo);

        std::string groupName;
        std::string dataSetName;
        std::tie(groupName, dataSetName) = getNames("", "", camDescr);

        // Check if file exists. If not, exit.
        if(m_featureBase->m_kernel->exists(groupName))
        {
            std::cout << timestamp << "ScanPositionIO: Loading camera " 
                      << groupName << "/" << dataSetName << std::endl;
            ScanCameraPtr cam = m_scanCameraIO->loadScanCamera(scanPosNo, camNo);
            ret->cams.push_back(cam);
        }
        else
        {
            break;
        }
        ++camNo;
    } while (true);

    // Get hyperspectral data
    Description hyperDescr = m_featureBase->m_description->hyperspectralCamera(scanPosNo);
    if(hyperDescr.dataSetName)
    {
        std::string groupName;
        std::string dataSetName;
        std::tie(groupName, dataSetName) = getNames("", "", hyperDescr);

        if (m_featureBase->m_kernel->exists(groupName))
        {
            std::cout << timestamp << "ScanPositionIO: Loading hyperspectral data... " << std::endl;
            HyperspectralCameraPtr hspCam = m_hyperspectralCameraIO->loadHyperspectralCamera(scanPosNo);
            ret->hyperspectralCamera = hspCam;
        }
    }

    return ret;
}

template <typename  FeatureBase>
bool ScanPositionIO< FeatureBase>::isScanPosition(const std::string& group) const
{
   return true;
}

} // namespace lvr2
