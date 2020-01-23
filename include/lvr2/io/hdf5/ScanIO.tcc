namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
void ScanIO<Derived>::save(std::string name, const ScanPtr& scanPtr)
{
    HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, name);
    save(g, scanPtr);
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

    // saving phi
    floatArr phi(new float[2]);
    phi[0] = scanPtr->phiMin;
    phi[1] = scanPtr->phiMax;
    m_arrayIO->save(group, "phi", dim, chunks, phi);

    // saving theta
    floatArr theta(new float[2]);
    theta[0] = scanPtr->thetaMin;
    theta[1] = scanPtr->thetaMax;
    m_arrayIO->save(group, "theta", dim, chunks, theta);

    // saving resolution
    floatArr resolution(new float[2]);
    resolution[0] = scanPtr->hResolution;
    resolution[1] = scanPtr->vResolution;
    m_arrayIO->save(group, "resolution", dim, chunks, resolution);

    // saving timestamps
    floatArr timestamp(new float[2]);
    timestamp[0] = scanPtr->startTime;
    timestamp[1] = scanPtr->endTime;
    m_arrayIO->save(group, "timestamps", dim, chunks, timestamp);
}

template <typename Derived>
ScanPtr ScanIO<Derived>::load(std::string name)
{
    ScanPtr ret;

    if (hdf5util::exist(m_file_access->m_hdf5_file, name))
    {
        HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, name, false);
        ret = load(g);
    }

    return ret;
}

template <typename Derived>
ScanPtr ScanIO<Derived>::loadScan(std::string name)
{
    return load(name);
}

template <typename Derived>
ScanPtr ScanIO<Derived>::load(HighFive::Group& group)
{
    ScanPtr ret(new Scan());

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
        }
    }

    // read poseEstimation
    boost::optional<lvr2::Transformd> poseEstimation =
        m_matrixIO->template load<lvr2::Transformd>(group, "poseEstimation");
    if (poseEstimation)
    {
        ret->poseEstimation = poseEstimation.get();
    }

    // read registration
    boost::optional<lvr2::Transformd> registration =
        m_matrixIO->template load<lvr2::Transformd>(group, "registration");
    if (registration)
    {
        ret->registration = registration.get();
    }

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

    // read phi
    if (group.exist("phi"))
    {
        std::vector<size_t> dimension;
        floatArr phi = m_arrayIO->template load<float>(group, "phi", dimension);

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

    // read theta
    if (group.exist("theta"))
    {
        std::vector<size_t> dimension;
        floatArr theta = m_arrayIO->template load<float>(group, "theta", dimension);

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

    // read resolution
    if (group.exist("resolution"))
    {
        std::vector<size_t> dimension;
        floatArr resolution = m_arrayIO->template load<float>(group, "resolution", dimension);

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

    // read timestamps
    if (group.exist("timestamps"))
    {
        std::vector<size_t> dimension;
        floatArr timestamps = m_arrayIO->template load<float>(group, "timestamps", dimension);

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
