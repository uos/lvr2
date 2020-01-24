namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
void ScanPositionIO<Derived>::save(uint scanPos, const ScanPositionPtr& scanPositionPtr)
{
    // TODO call save with group
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
        std::cout << "path" << std::endl;

        m_scanIO->template save(scanGroup, scanPtr);
    }

    // saving contained scans
    pos = 0;
    for (ScanCameraPtr scanCameraPtr : scanPositionPtr->cams)
    {
        // char buffer[sizeof(int) * 5];
        // sprintf(buffer, "%05d", pos++);
        // string nr_str(buffer);

        // std::string path = "/scans/data/" + nr_str;
        // std::cout << path << std::endl;

        // HighFive::Group scanGroup = hdf5util::getGroup(group, path);
        // std::cout << "path" << std::endl;

        // m_scanIO->template save(scanGroup, scanPtr);
    }

    // set dim and chunks for gps position
    std::vector<size_t> dim{2, 1};
    std::vector<hsize_t> chunks{2, 1};

    // saving gps position
    doubleArr gpsPosition(new double[2]);
    gpsPosition[0] = scanPositionPtr->latitude;
    gpsPosition[1] = scanPositionPtr->longitude;
    m_arrayIO->save(group, "gpsPosition", dim, chunks, gpsPosition);

    // saving estimated and registrated pose
    m_matrixIO->save(group, "poseEstimate", scanPositionPtr->poseEstimate);
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

    // TODO: create prefix from scanPos

    if (hdf5util::exist(m_file_access->m_hdf5_file, scanPos))
    {
        HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, scanPos, false);
        ret = load(g);
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
            ScanPtr scan = m_scanIO->template load(g);
            std::cout << "  loadded scan" << std::endl;

            ret->scans.push_back(scan);
            std::cout << "  added scan" << std::endl;
        }
    }

    std::cout << "  loading gps position" << std::endl;

    // read gpsPosition
    if (group.exist("gpsPosition"))
    {
        std::vector<size_t> dimension;
        doubleArr gpsPosition = m_arrayIO->template load<double>(group, "gpsPosition", dimension);

        if (dimension.at(0) != 2)
        {
            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong gps position dimension. The gps "
                         "position will not be loaded."
                      << std::endl;
        }
        else
        {
            ret->latitude = gpsPosition[0];
            ret->longitude = gpsPosition[1];
        }
    }

    std::cout << "  loading poseEstimation" << std::endl;

    // read poseEstimation
    boost::optional<lvr2::Transformd> poseEstimate =
        m_matrixIO->template load<lvr2::Transformd>(group, "poseEstimate");
    if (poseEstimate)
    {
        ret->poseEstimate = poseEstimate.get();
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
