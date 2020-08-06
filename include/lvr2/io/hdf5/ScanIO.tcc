namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
void ScanIO<Derived>::save(uint scanPos, uint scanNr, const ScanPtr& scanPtr)
{
    char bufferPos[sizeof(int) * 5];
    sprintf(bufferPos, "%08d", scanPos);
    string pos_str(bufferPos);

    char bufferScan[sizeof(int) * 5];
    sprintf(bufferScan, "%08d", scanNr);
    string nr_str(bufferScan);

    std::string basePath = "raw/" + pos_str + "/scans/data/" + nr_str;

    HighFive::Group scanGroup = hdf5util::getGroup(m_file_access->m_hdf5_file, basePath);

    save(scanGroup, scanPtr);
}

template <typename Derived>
void ScanIO<Derived>::save(HighFive::Group& group, uint scanNr, const ScanPtr& scanPtr)
{
    char bufferScan[sizeof(int) * 5];
    sprintf(bufferScan, "%08d", scanNr);
    string nr_str(bufferScan);

    std::string basePath = "/scans/data/" + nr_str;

    HighFive::Group scanGroup = hdf5util::getGroup(group, basePath);

    save(scanGroup, scanPtr);
}

template <typename Derived>
void ScanIO<Derived>::save(HighFive::Group& group, const ScanPtr& scanPtr)
{
    std::string id(ScanIO<Derived>::ID);
    std::string obj(ScanIO<Derived>::OBJID);
    hdf5util::setAttribute(group, "IO", id);
    hdf5util::setAttribute(group, "CLASS", obj);

    // save points
    std::vector<size_t> scanDim = {scanPtr->points->numPoints(), 3};
    std::vector<hsize_t> scanChunk = {scanPtr->points->numPoints(), 3};
    boost::shared_array<float> points = scanPtr->points->getPointArray();
    m_arrayIO->template save<float>(group, "points", scanDim, scanChunk, points);

    // saving estimated and registrated pose
    m_matrixIO->save(group, "poseEstimation", scanPtr->poseEstimation);
    m_matrixIO->save(group, "registration", scanPtr->registration);

    // set dim and chunks for boundingBox
    std::vector<size_t> dim{2, 3};
    std::vector<hsize_t> chunks{2, 3};

    // create and save boundingBox
    floatArr bbox(new float[6]);
    bbox[0] = scanPtr->boundingBox.getMin()[0];
    bbox[1] = scanPtr->boundingBox.getMin()[1];
    bbox[2] = scanPtr->boundingBox.getMin()[2];
    bbox[3] = scanPtr->boundingBox.getMax()[0];
    bbox[4] = scanPtr->boundingBox.getMax()[1];
    bbox[5] = scanPtr->boundingBox.getMax()[2];
    m_arrayIO->save(group, "boundingBox", dim, chunks, bbox);

    // set dim and chunks for saving following data
    dim = {2, 1};
    chunks = {2, 1};

    // saving theta
    doubleArr theta(new double[2]);
    theta[0] = scanPtr->thetaMin;
    theta[1] = scanPtr->thetaMax;
    m_arrayIO->save(group, "theta", dim, chunks, theta);

    // saving phi
    doubleArr phi(new double[2]);
    phi[0] = scanPtr->phiMin;
    phi[1] = scanPtr->phiMax;
    m_arrayIO->save(group, "phi", dim, chunks, phi);

    // saving resolution
    doubleArr resolution(new double[2]);
    resolution[0] = scanPtr->hResolution;
    resolution[1] = scanPtr->vResolution;
    m_arrayIO->save(group, "resolution", dim, chunks, resolution);

    // saving timestamps
    doubleArr timestamp(new double[2]);
    timestamp[0] = scanPtr->startTime;
    timestamp[1] = scanPtr->endTime;
    m_arrayIO->save(group, "timestamps", dim, chunks, timestamp);
}

template <typename Derived>
ScanPtr ScanIO<Derived>::load(uint scanPos, uint scanNr)
{
    ScanPtr ret;

    char scan_buffer[sizeof(int) * 5];
    sprintf(scan_buffer, "%08d", scanPos);
    string scanPos_str(scan_buffer);

    char buffer[sizeof(int) * 5];
    sprintf(buffer, "%08d", scanNr);
    string nr_str(buffer);

    std::string basePath = "raw/" + scanPos_str + "/scans/data" + nr_str;

    if (hdf5util::exist(m_file_access->m_hdf5_file, basePath))
    {
        HighFive::Group group = hdf5util::getGroup(m_file_access->m_hdf5_file, basePath);
        ret = load(group);
    }

    return ret;
}

template <typename Derived>
ScanPtr ScanIO<Derived>::load(HighFive::Group& group, uint scanNr)
{
    ScanPtr ret;

    char buffer[sizeof(int) * 5];
    sprintf(buffer, "%08d", scanNr);
    string nr_str(buffer);

    std::string basePath = "/scans/data" + nr_str;

    if (hdf5util::exist(group, basePath))
    {
        HighFive::Group g = hdf5util::getGroup(group, basePath);
        ret = load(g);
    }

    return ret;
}

// template <typename Derived>
// ScanPtr ScanIO<Derived>::loadScan(HighFive::Group& group, std::string name)
// {
//     ScanPtr ret;
//     HighFive::Group g = hdf5util::getGroup(group, name, false);
//     ret = load(g);
//     return ret;
// }

template <typename Derived>
ScanPtr ScanIO<Derived>::load(HighFive::Group& group)
{
    ScanPtr ret(new Scan());

    if (!isScan(group))
    {
        std::cout << "[Hdf5IO - ScanIO] WARNING: flags of " << group.getId() << " are not correct."
                  << std::endl;
        return ret;
    }

    std::cout << "    loading points" << std::endl;

    // read points
    if (group.exist("points"))
    {
        std::vector<size_t> dimension;
        floatArr pointArr = m_arrayIO->template load<float>(group, "points", dimension);

        if (dimension[1] != 3)
        {
            std::cout
                << "[Hdf5IO - ScanIO] WARNING: Wrong point dimensions. Points will not be loaded."
                << std::endl;
        }
        else
        {
            ret->points = PointBufferPtr(new PointBuffer(pointArr, dimension[0]));
            ret->numPoints = dimension[0];
            ret->pointsLoaded = true;
        }
    }

    std::cout << "    loading poseEstimations" << std::endl;
    // read poseEstimation
    boost::optional<lvr2::Transformd> poseEstimation =
        m_matrixIO->template load<lvr2::Transformd>(group, "poseEstimation");
    if (poseEstimation)
    {
        ret->poseEstimation = poseEstimation.get();
    }

    std::cout << "    loading registration" << std::endl;
    // read registration
    boost::optional<lvr2::Transformd> registration =
        m_matrixIO->template load<lvr2::Transformd>(group, "registration");
    if (registration)
    {
        ret->registration = registration.get();
    }

    std::cout << "    loading boundingBox" << std::endl;
    // read boundingBox
    if (group.exist("boundingBox"))
    {
        std::vector<size_t> dimension;
        floatArr bb = m_arrayIO->template load<float>(group, "boundingBox", dimension);

        if ((dimension[0] != 2 || dimension[1] != 3) && dimension[0] != 6)
        {
            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong boundingBox dimensions. BoundingBox "
                         "will not be loaded."
                      << std::endl;
        }
        else
        {
            BaseVector<float> bb_min(bb[0], bb[1], bb[2]);
            BaseVector<float> bb_max(bb[3], bb[4], bb[5]);
            BoundingBox<BaseVector<float>> boundingBox(bb_min, bb_max);
            ret->boundingBox = boundingBox;
        }
    }

    std::cout << "    loading theta" << std::endl;
    // read theta
    if (group.exist("theta"))
    {
        std::vector<size_t> dimension;
        doubleArr theta = m_arrayIO->template load<double>(group, "theta", dimension);

        if (dimension.at(0) != 2)
        {
            std::cout
                << "[Hdf5IO - ScanIO] WARNING: Wrong theta dimension. Theta will not be loaded."
                << std::endl;
        }
        else
        {
            ret->thetaMin = theta[0];
            ret->thetaMax = theta[1];
        }
    }

    std::cout << "    loading phi" << std::endl;
    // read phi
    if (group.exist("phi"))
    {
        std::vector<size_t> dimension;
        doubleArr phi = m_arrayIO->template load<double>(group, "phi", dimension);

        if (dimension[0] != 2)
        {
            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong phi dimension. Phi will not be loaded."
                      << std::endl;
        }
        else
        {
            ret->phiMin = phi[0];
            ret->phiMax = phi[1];
        }
    }

    std::cout << "    loading resolution" << std::endl;
    // read resolution
    if (group.exist("resolution"))
    {
        std::vector<size_t> dimension;
        doubleArr resolution = m_arrayIO->template load<double>(group, "resolution", dimension);

        if (dimension[0] != 2)
        {
            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong resolution dimensions. Resolution will "
                         "not be loaded."
                      << std::endl;
        }
        else
        {
            ret->hResolution = resolution[0];
            ret->vResolution = resolution[1];
        }
    }

    std::cout << "    loading timestamps" << std::endl;
    // read timestamps
    if (group.exist("timestamps"))
    {
        std::vector<size_t> dimension;
        doubleArr timestamps = m_arrayIO->template load<double>(group, "timestamps", dimension);

        if (dimension[0] != 2)
        {
            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong timestamp dimensions. Timestamp will "
                         "not be loaded."
                      << std::endl;
        }
        else
        {
            ret->startTime = timestamps[0];
            ret->endTime = timestamps[1];
        }
    }

    return ret;
}

template <typename Derived>
bool ScanIO<Derived>::isScan(HighFive::Group& group)
{
    std::string id(ScanIO<Derived>::ID);
    std::string obj(ScanIO<Derived>::OBJID);
    return hdf5util::checkAttribute(group, "IO", id) &&
           hdf5util::checkAttribute(group, "CLASS", obj);
}

} // namespace hdf5features

} // namespace lvr2
