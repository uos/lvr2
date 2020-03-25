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

    char buffer[sizeof(int) * 5];
    sprintf(buffer, "%08d", scanPos);
    string nr_str(buffer);
    std::string basePath = "raw/" + nr_str + "/";

    if (hdf5util::exist(m_file_access->m_hdf5_file, basePath))
    {
        HighFive::Group group = hdf5util::getGroup(m_file_access->m_hdf5_file, basePath);
        ret = load(group);
    }

    return ret;
}

template <typename  FeatureBase>
ScanPositionPtr ScanPositionIO< FeatureBase>::load(const std::string& group, const std::string& container)
{
    ScanPositionPtr ret;

    char buffer[sizeof(int) * 5];
    sprintf(buffer, "%08d", scanPos);
    string nr_str(buffer);
    std::string basePath = "raw/" + nr_str + "/";

    if (hdf5util::exist(m_file_access->m_hdf5_file, basePath))
    {
        HighFive::Group group = hdf5util::getGroup(m_file_access->m_hdf5_file, basePath);
        ret = load(group);
    }

    return ret;
}

template <typename  FeatureBase>
ScanPositionPtr ScanPositionIO< FeatureBase>::load(HighFive::Group& group)
{
    ScanPositionPtr ret(new ScanPosition);

    if (!isScanPosition(group))
    {
        std::cout << "[Hdf5IO - ScanPositionIO] WARNING: flags of " << group.getId()
                  << " are not correct." << std::endl;
        return ret;
    }

    std::cout << "  loading scans" << std::endl;

    // ret(new ScanPosition);

    // load all scans
    HighFive::Group hfscans = hdf5util::getGroup(group, "/scans/data");
    for (std::string groupname : hfscans.listObjectNames())
    {
        std::cout << "  " << groupname << std::endl;

        if (hdf5util::exist(hfscans, groupname))
        {
            HighFive::Group g = hdf5util::getGroup(hfscans, "/" + groupname);
            std::cout << "  try to load scan" << std::endl;
            ScanPtr scan = m_scanIO->load(g);
            std::cout << "  loadded scan" << std::endl;

            ret->scans.push_back(scan);
            std::cout << "  added scan" << std::endl;
        }
    }

    for (std::string groupname : group.listObjectNames())
    {
        // load all scanCameras
        if (std::regex_match(groupname, std::regex("cam_\\d{2}")))
        {
            HighFive::Group g = hdf5util::getGroup(group, "/" + groupname);
            ScanCameraPtr scanCameraPtr = m_scanCameraIO->load(g);
            ret->cams.push_back(scanCameraPtr);
        }
    }

    // load hyperspectralCamera
    std::string path = "spectral";
    std::cout << "  loading spectral camera" << std::endl;
    if (group.exist(path))
    {
        HighFive::Group spectralGroup = hdf5util::getGroup(group, "/" + path);
        if (spectralGroup.exist("data"))
        {
            spectralGroup = hdf5util::getGroup(spectralGroup, "/data");
            HyperspectralCameraPtr ptr = m_hyperspectralCameraIO->load(spectralGroup);
            ret->hyperspectralCamera = ptr;
        }
    }

    std::cout << "  loading gps position" << std::endl;

    // read gpsPosition
    if (group.exist("gpsPosition"))
    {
        std::vector<size_t> dimension;
        doubleArr gpsPosition = m_arrayIO->template load<double>(group, "gpsPosition", dimension);

        if (dimension.at(0) != 3)
        {
            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong gps position dimension. The gps "
                         "position will not be loaded."
                      << std::endl;
        }
        else
        {
            ret->latitude = gpsPosition[0];
            ret->longitude = gpsPosition[1];
            ret->altitude = gpsPosition[2];
        }
    }

    std::cout << "  loading poseEstimation" << std::endl;

    // read poseEstimation
    boost::optional<lvr2::Transformd> pose_estimate =
        m_matrixIO->template load<lvr2::Transformd>(group, "pose_estimate");
    if (pose_estimate)
    {
        ret->pose_estimate = pose_estimate.get();
    }

    // read registration
    boost::optional<lvr2::Transformd> registration =
        m_matrixIO->template load<lvr2::Transformd>(group, "registration");
    if (registration)
    {
        ret->registration = registration.get();
    }

    // read timestamp
    if (group.exist("timestamp"))
    {
        std::vector<size_t> dimension;
        doubleArr timestamp = m_arrayIO->template load<double>(group, "timestamp", dimension);

        if (dimension.at(0) != 1)
        {
            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong timestamp dimension. The timestamp will "
                         "not be loaded."
                      << std::endl;
        }
        else
        {
            ret->timestamp = timestamp[0];
        }
    }

    return ret;
}

template <typename  FeatureBase>
bool ScanPositionIO< FeatureBase>::isScanPosition(HighFive::Group& group)
{
   return true;
}

} // namespace lvr2
