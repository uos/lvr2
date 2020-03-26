namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
void ScanPositionIO<Derived>::save(uint scanPos, const ScanPositionPtr& scanPositionPtr)
{
    char buffer[sizeof(int) * 5];
    sprintf(buffer, "%08d", scanPos);
    string nr_str(buffer);

    std::string basePath = "raw/" + nr_str;

    HighFive::Group scanPosGroup = hdf5util::getGroup(m_file_access->m_hdf5_file, basePath);

    save(scanPosGroup, scanPositionPtr);
}

template <typename Derived>
void ScanPositionIO<Derived>::save(HighFive::Group& group, const ScanPositionPtr& scanPositionPtr)
{
    std::string id(ScanPositionIO<Derived>::ID);
    std::string obj(ScanPositionIO<Derived>::OBJID);
    hdf5util::setAttribute(group, "IO", id);
    hdf5util::setAttribute(group, "CLASS", obj);

    // saving contained scans
    int pos = 0;
    for (ScanPtr scanPtr : scanPositionPtr->scans)
    {
        char buffer[sizeof(int) * 5];
        sprintf(buffer, "%08d", pos++);
        string nr_str(buffer);

        std::string path = "/scans/data/" + nr_str;
        std::cout << path << std::endl;

        HighFive::Group scanGroup = hdf5util::getGroup(group, path);

        m_scanIO->save(scanGroup, scanPtr);
    }

    // saving contained cameras
    pos = 0;
    for (ScanCameraPtr scanCameraPtr : scanPositionPtr->cams)
    {
        char buffer[sizeof(int) * 5];
        sprintf(buffer, "%02d", pos++);
        string nr_str(buffer);

        std::string path = "/cam_" + nr_str;
        std::cout << "saving camera " << (pos - 1) << " at " << path << std::endl;

        HighFive::Group camGroup = hdf5util::getGroup(group, path);

        m_scanCameraIO->save(camGroup, scanCameraPtr);
    }

    // saving hyperspectral camera
    if (scanPositionPtr->hyperspectralCamera)
    {
        std::string path = "/spectral/data";
        std::cout << "  saving spectral camera" << std::endl;
        HighFive::Group camGroup = hdf5util::getGroup(group, path);
        m_hyperspectralCameraIO->save(camGroup, scanPositionPtr->hyperspectralCamera);
    }

    // set dim and chunks for gps position
    std::vector<size_t> dim{3, 1};
    std::vector<hsize_t> chunks{3, 1};

    // saving gps position
    doubleArr gpsPosition(new double[3]);
    gpsPosition[0] = scanPositionPtr->latitude;
    gpsPosition[1] = scanPositionPtr->longitude;
    gpsPosition[2] = scanPositionPtr->altitude;
    m_arrayIO->save(group, "gpsPosition", dim, chunks, gpsPosition);

    dim = {2, 1};
    chunks = {2, 1};

    // saving estimated and registrated pose
    m_matrixIO->save(group, "pose_estimate", scanPositionPtr->pose_estimate);
    m_matrixIO->save(group, "registration", scanPositionPtr->registration);

    // set dim and chunks for timestamp
    dim = {1, 1};
    chunks = {1, 1};

    // saving timestamp
    doubleArr timestamp(new double[1]);
    timestamp[0] = scanPositionPtr->timestamp;
    m_arrayIO->save(group, "timestamp", dim, chunks, timestamp);
}

template <typename Derived>
ScanPositionPtr ScanPositionIO<Derived>::load(uint scanPos)
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

template <typename Derived>
ScanPositionPtr ScanPositionIO<Derived>::loadScanPosition(uint scanPos)
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

template <typename Derived>
ScanPositionPtr ScanPositionIO<Derived>::load(HighFive::Group& group)
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

template <typename Derived>
bool ScanPositionIO<Derived>::isScanPosition(HighFive::Group& group)
{
    std::string id(ScanPositionIO<Derived>::ID);
    std::string obj(ScanPositionIO<Derived>::OBJID);
    return hdf5util::checkAttribute(group, "IO", id) &&
           hdf5util::checkAttribute(group, "CLASS", obj);
}

} // namespace hdf5features

} // namespace lvr2
