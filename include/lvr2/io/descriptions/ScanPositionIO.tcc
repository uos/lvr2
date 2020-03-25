namespace lvr2
{

template <typename  FeatureBase>
void ScanPositionIO< FeatureBase>::save(const size_t& scanPosNo, const ScanPositionPtr& scanPositionPtr)
{
    Description d = m_featureBase->m_description->position(scanPosNo);
  
    // Setup defaults
    std::stringstream sstr;
    sstr << std::setfill('0') << std::setw(8) << scanPosNo;

    std::string metaName = "meta.yaml";
    std::string groupName = sstr().str();
   
    if(d.metaName)
    {
        metaName = d.metaName;
    }

    if(d.groupName)
    {
        groupName = d.groupName;
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
                 
        YAML::Node node = *scanPositionPtr;
        m_featureBase->m_kernel->saveMetaYAML(groupName, metaName, node);
    }
    
    // Save all scans
    for(size_t i = 0; i < scanPositionPtr->scans.size(); i++)
    {
        m_scanIO->save(scanPosNo, i, scanPositionPtr->scans[i]);
    }

    // Save all scan camera and images
    for(size_t i = 0; i < scanPositionPtr->cams.size(); i++)
    {
        m_scanCameraIO->save(scanPosNo, i, scanPositionPtr->cams[i]);
    }
    
    // Save hyperspectral data
    if (scanPositionPtr->hyperspectralCamera)
    {
        m_hyperspectralCameraIO->save(scanPosNo, scanPositionPtr->hyperspectralCamera);
    }
}

template <typename  FeatureBase>
ScanPositionPtr ScanPositionIO< FeatureBase>::load(const size_t& scanPosNo)
{
    ScanPositionPtr ret;

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
    std::string groupName = sstr().str();

    if(d.metaName)
    {
        metaName = d.metaName;
    }

    if(d.groupName)
    {
        groupName = d.groupName;
    }

    if(!d.metaData)
    {
        std::cout << timestamp << "ScanPositionIO::load(): Warning: No meta information "
                  << "for scan position " << scanPosNo << " found." << std::endl;
        std::cout << timestamp << "Creating new meta data with default values." << std::endl; 
        d.metaData = *ret;
    }
    else
    {
        ret = *(d.metaData);
    }
    
    // Get all sub scans
    size_t scanNo = 0;
    do
    {
        // Get description for next scan
        Description scanDescr = m_featureBase->m_description->scan(scanPosNo, scanNo);

        // Check if it exists. If not, exit.
        if(m_featureBase->m_kernel->exists(scanDescr.groupName, scanDescr.dataSetName))
        {
            std::cout << timestamp << "ScanPositionIO: Loading scan " 
                      << scanDescr.groupName << "/" << scanDescr.dataSetName << std::endl;
            ScanPtr scan = m_scanIO->load(scanPosNo, scanNo);
            ret->scans->push_back(scan);
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

        // Check if it exists. If not, exit.
        if(m_featureBase->m_kernel->exists(camDescr.groupName, camDescr.dataSetName))
        {
            std::cout << timestamp << "ScanPositionIO: Loading camera " 
                      << camDescr.groupName << "/" << camDescr.dataSetName << std::endl;
            ScanCameraPtr cam = m_scanCameraIO->load(scanPosNo, scanNo);
            ret->cams->push_back(scan);
        }
        else
        {
            break;
        }
        ++camNo;
    } while (true);

    // Get hyperspectral data
    Description hyperDescr = m_featureBase->m_decription->hyperSpectralCamera(scanPosNo);
    if(m_featureBase->m_kernel->exists(camDescr.groupName)
    {
        std::cout << timestamp << "ScanPositionIO: Loading hyperspectral data... " << std::endl;
        HyperspectralCameraPtr hspCam = m_hyperspectralCameraIO->load(scanPosNo);
        ret->hyperspectralCamera = hspCam;
    }

    return ret;
}


template <typename  FeatureBase>
bool ScanPositionIO< FeatureBase>::isScanPosition(HighFive::Group& group)
{
   return true;
}

} // namespace lvr2
