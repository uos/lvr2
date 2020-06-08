#include "lvr2/io/yaml/ScanPosition.hpp"

 #include <boost/optional/optional_io.hpp>

namespace lvr2
{

template <typename  FeatureBase>
void ScanPositionIO< FeatureBase>::saveScanPosition(const size_t& scanPosNo, const ScanPositionPtr& scanPositionPtr)
{
    Description d = m_featureBase->m_description->position(scanPosNo);
  
    // Setup defaults
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanPosNo;

    std::string metaName = "meta.yaml";
    std::string groupName = sstr.str();
   
    if(d.metaName)
    {
        metaName = *d.metaName;
    }

    if(d.groupName)
    {
        groupName = *d.groupName;
    }

    // Save meta information
    if(d.metaData)
    {
        m_featureBase->m_kernel->saveMetaYAML(groupName, metaName, *(d.metaData));
    }
    else
    {
        std::cout << timestamp << "ScanPositionIO::save(): Warning: No meta information "
                  << "for scan position " << scanPosNo << " found." << std::endl;
        std::cout << timestamp << "Creating new meta data from given struct." << std::endl; 
                 
        YAML::Node node;
        node = *scanPositionPtr;
        m_featureBase->m_kernel->saveMetaYAML(groupName, metaName, node);
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
ScanPositionPtr ScanPositionIO< FeatureBase>::loadScanPosition(const size_t& scanPosNo)
{
    ScanPositionPtr ret(new ScanPosition);

    // char buffer[sizeof(int) * 5];
    // sprintf(buffer, "%08d", scanPos);
    // string nr_str(buffer);
    // std::string basePath = "raw/" + nr_str + "/";

    // if (hdf5util::exist(m_file_access->m_hdf5_file, basePath))
    // {
    //     HighFive::Group group = hdf5util::getGroup(m_file_access->m_hdf5_file, basePath);
    //     ret = load(group);
    // }

    Description d = m_featureBase->m_description->position(scanPosNo);

    // Setup defaults
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanPosNo;

    std::string metaName = "meta.yaml";
    std::string groupName = sstr.str();

    if(d.metaName)
    {
        metaName = *d.metaName;
    }

    if(d.groupName)
    {
        groupName = *d.groupName;
    }

    if(!d.metaData)
    {
        std::cout << timestamp << "ScanPositionIO::load(): Warning: No meta information "
                  << "for scan position " << scanPosNo << " found." << std::endl;
        std::cout << timestamp << "Creating new meta data with default values." << std::endl; 
        YAML::Node node;
        node = *ret;
        d.metaData = node;
    }
    else
    {
        *ret = (*d.metaData).as<ScanPosition>();
    }
    
    // Get all sub scans
    size_t scanNo = 0;
    do
    {
        // Get description for next scan
        Description scanDescr = m_featureBase->m_description->scan(scanPosNo, scanNo);

        std::string groupName;
        std::string dataSetName;
        std::tie(groupName, dataSetName) = getNames("", "", scanDescr);

        // Check if it exists. If not, exit.
        if(m_featureBase->m_kernel->exists(groupName, dataSetName))
        {
            std::cout << timestamp << "ScanPositionIO: Loading scan " 
                      << groupName << "/" << dataSetName << std::endl;
            ScanPtr scan = m_scanIO->loadScan(scanPosNo, scanNo);
            ret->scans.push_back(scan);
            std::cout << scan->points->numPoints() << std::endl;
        }
        else
        {
            break;
        }
        ++scanNo;
    } 
    while (true);

    // Get all scan camera
    size_t camNo = 0;
    do
    {
        // Get description for next scan
        Description camDescr = m_featureBase->m_description->scanCamera(scanPosNo, scanNo);

        std::string groupName;
        std::string dataSetName;
        std::tie(groupName, dataSetName) = getNames("", "", camDescr);

        // Check if file exists. If not, exit.
        if(m_featureBase->m_kernel->exists(groupName, dataSetName))
        {
            std::cout << timestamp << "ScanPositionIO: Loading camera " 
                      << groupName << "/" << dataSetName << std::endl;
            ScanCameraPtr cam = m_scanCameraIO->loadScanCamera(scanPosNo, scanNo);
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
bool ScanPositionIO< FeatureBase>::isScanPosition(const std::string& group)
{
   return true;
}

} // namespace lvr2
