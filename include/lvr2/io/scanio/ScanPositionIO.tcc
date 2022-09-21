namespace lvr2
{

namespace scanio
{

template <typename  BaseIO>
bool ScanPositionIO< BaseIO>::save(
    const size_t& scanPosNo, 
    ScanPositionPtr scanPositionPtr) const
{
    Description d = m_baseIO->m_description->position(scanPosNo);
  
    // std::cout << "[ScanPositionIO] ScanPosition " << scanPosNo << " - Description: " << std::endl;
    // std::cout << d << std::endl;

    if(!d.dataRoot)
    {
        // dataRoot has to be specified!
        return false;
    }

    if(!scanPositionPtr)
    {
        return false;
    }

    // Save all lidar sensors

    for(size_t i = 0; i < scanPositionPtr->lidars.size(); i++)
    {

        // std::cout << " [ScanPositionIO]: Writing lidar " << i << std::endl;
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

    // std::cout << "[ScanPositionIO] Hyper written. " << std::endl;

    // std::cout << "[ScanPositionIO - save] Write Meta." << std::endl;
    // Save meta information
    if(d.meta)
    {
        YAML::Node node;
        node = *scanPositionPtr;
        m_baseIO->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
    }
    return true;
    // std::cout << "[ScanPositionIO] Meta written. " << std::endl;
}

template <typename BaseIO>
boost::optional<YAML::Node> ScanPositionIO<BaseIO>::loadMeta(
        const size_t& scanPosNo) const
{
    Description d = m_baseIO->m_description->position(scanPosNo);
    return m_metaIO->load(d);
}

template <typename BaseIO>
ScanPositionPtr ScanPositionIO<BaseIO>::load(
    const size_t& scanPosNo) const
{
    ScanPositionPtr ret;

    Description d = m_baseIO->m_description->position(scanPosNo);

    // std::cout << "[ScanPositionIO - load]"  << std::endl;
    // std::cout << d <<  std::endl;

    if(!d.dataRoot)
    {
        return ret;
    }

    // Check if specified scan position exists
    if(!m_baseIO->m_kernel->exists(*d.dataRoot))
    {
        return ret;
    }

    if(d.meta)
    {
        YAML::Node meta;
        cout<<"Meta laden" << scanPosNo<< endl;
        if(!m_baseIO->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
        {
            cout << "empty" <<endl;
            return ret;
        }

        try {

            ret = std::make_shared<ScanPosition>(meta.as<ScanPosition>());

        } catch(const YAML::TypedBadConversion<ScanPosition>& ex) {
            std::cerr << "[ScanPositionIO - load] ERROR at Scan (" << scanPosNo << ") : Could not decode YAML as ScanPosition." << std::endl;
            throw ex;
        }
        cout<<"Meta laden--" << scanPosNo<< endl;
       
    } else {
        // no meta name specified but scan position is there: 
        ret = std::make_shared<ScanPosition>();
    }

    //// DATA
    // std::cout << "[ScanPositionIO - load] Load SENSORS " << std::endl;

    // Get all lidar sensors
    size_t lidarNo = 0;
    while(true)
    {
        // std::cout << "[ScanPositionIO - load] Load LIDAR " << lidarNo << std::endl;
        LIDARPtr lidar = m_lidarIO->load(scanPosNo, lidarNo);

        if(lidar)
        {
            ret->lidars.push_back(lidar);
        } else {
            break;
        }

        // std::cout << "[ScanPositionIO - load] Loaded LIDAR " << lidarNo << std::endl;

        ++lidarNo;
    }

    // Get all scan cameras
    size_t camNo = 0;
    while(true)
    {
        // std::cout << "[ScanPositionIO - load] Load Camera " << camNo << std::endl;
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
            // HERE
            ret->hyperspectral_cameras.push_back(cam);
        } else {
            break;
        }

        // std::cout << "[ScanPositionIO - load] Loaded HyperCamera " << hCamNo << std::endl;

        hCamNo++;
    }

    return ret;
}

template <typename  BaseIO>
ScanPositionPtr ScanPositionIO< BaseIO>::load(
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

template <typename  BaseIO>
bool ScanPositionIO< BaseIO>::saveScanPosition(
    const size_t& scanPosNo, 
    ScanPositionPtr scanPositionPtr) const
{
   return save(scanPosNo, scanPositionPtr);
}

template <typename  BaseIO>
ScanPositionPtr ScanPositionIO<BaseIO>::loadScanPosition(
    const size_t& scanPosNo) const
{
    return load(scanPosNo);
}

template <typename  BaseIO>
ScanPositionPtr ScanPositionIO< BaseIO>::loadScanPosition(
    const size_t& scanPosNo, ReductionAlgorithmPtr reduction) const
{
    return load(scanPosNo, reduction);
}

} // namespace scanio

} // namespace lvr2
