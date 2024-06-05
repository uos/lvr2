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

    lvr2::logout::get() << lvr2::info << "[ScanPositionIO] ScanPosition " << scanPosNo << " - Description: " << lvr2::endl;
    lvr2::logout::get() << lvr2::info << d;

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
    //cout <<scanPositionPtr->lidars.size()<< endl;
    for(size_t i = 0; i < scanPositionPtr->lidars.size(); i++)
    {

        // lvr2::logout::get() << " [ScanPositionIO]: Writing lidar " << i << lvr2::endl;
        m_lidarIO->save(scanPosNo, i, scanPositionPtr->lidars[i]);
    }

    // lvr2::logout::get() << "[ScanPositionIO] LIDARs written. " << lvr2::endl;

    // Save all scan camera sensors
    for(size_t i = 0; i < scanPositionPtr->cameras.size(); i++)
    {
        // lvr2::logout::get() << " [ScanPositionIO]: Writing camera " << i << lvr2::endl;
        m_cameraIO->save(scanPosNo, i, scanPositionPtr->cameras[i]);
    }

    // lvr2::logout::get() << "[ScanPositionIO] Cameras written. " << lvr2::endl;
    
    // Save all hyperspectral camera sensors
    for(size_t i=0; i < scanPositionPtr->hyperspectral_cameras.size(); i++)
    {
        m_hyperspectralCameraIO->save(scanPosNo, i, scanPositionPtr->hyperspectral_cameras[i]);
    }

    // lvr2::logout::get() << "[ScanPositionIO] Hyper written. " << lvr2::endl;

    // lvr2::logout::get() << "[ScanPositionIO - save] Write Meta." << lvr2::endl;
    // Save meta information
    if(d.meta)
    {
        YAML::Node node;
        node = *scanPositionPtr;
        m_baseIO->m_kernel->saveMetaYAML(*d.metaRoot, *d.meta, node);
    }
    return true;
    // lvr2::logout::get() << "[ScanPositionIO] Meta written. " << lvr2::endl;
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

    lvr2::logout::get() << "[ScanPositionIO - load]"  << lvr2::endl;
    lvr2::logout::get() << d <<  lvr2::endl;

    if(!d.dataRoot)
    {
        lvr2::logout::get() << lvr2::warning << "[ScanPositionIO - load] - Data root not specified." << lvr2::endl;
        return ret;
    }

    // Check if specified scan position exists
    if(!m_baseIO->m_kernel->exists(*d.dataRoot))
    {
        lvr2::logout::get() << lvr2::warning << "[ScanPositionIO - load] - Data root does not exist." << lvr2::endl;
        return ret;
    }

    if(d.meta)
    {
        YAML::Node meta;

        if(!m_baseIO->m_kernel->loadMetaYAML(*d.metaRoot, *d.meta, meta))
        {
            lvr2::logout::get() << lvr2::warning << "[ScanPositionIO - load] - Unable to load meta data." << lvr2::endl;
            //return ret;
        }

        meta["original_name"] = scanPosNo;

        try
        {

            ret = std::make_shared<ScanPosition>(meta.as<ScanPosition>());
        }
        catch (const YAML::TypedBadConversion<ScanPosition> &ex)
        {
            lvr2::logout::get() << lvr2::error << "[ScanPositionIO - Load] Scan (" << scanPosNo << ") : Could not decode YAML as ScanPosition." << lvr2::endl;
            throw ex;
        }
    }
    else
    {
        // no meta name specified but scan position is there:
        ret = std::make_shared<ScanPosition>();
    }

    //// DATA
    lvr2::logout::get() << "[ScanPositionIO - load] Load SENSORS " << lvr2::endl;

    // Get all lidar sensors
    size_t lidarNo = 0;
    while(true)
    {
        lvr2::logout::get() << "[ScanPositionIO - load] Load LIDAR " << lidarNo << lvr2::endl;
        LIDARPtr lidar = m_lidarIO->load(scanPosNo, lidarNo);

        if (lidar)
        {
            ret->lidars.push_back(lidar);
        }
        else
        {
            break;
        }

        lvr2::logout::get() << "[ScanPositionIO - load] Loaded LIDAR " << lidarNo << lvr2::endl;

        ++lidarNo;
    }

    // Get all scan cameras
    size_t camNo = 0;
    while(true)
    {
        lvr2::logout::get() << "[ScanPositionIO - load] Load Camera " << camNo << lvr2::endl;
        CameraPtr cam = m_cameraIO->load(scanPosNo, camNo);
        if (cam)
        {
            ret->cameras.push_back(cam);
        }
        else
        {
            break;
        }
        lvr2::logout::get() << "[ScanPositionIO - load] Loaded Camera " << camNo << lvr2::endl;
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

        lvr2::logout::get() << "[ScanPositionIO - load] Loaded HyperCamera " << hCamNo << lvr2::endl;

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
