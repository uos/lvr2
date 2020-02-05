namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
void HyperspectralCameraIO<Derived>::save(HighFive::Group& group,
                                          const HyperspectralCameraPtr& hyperspectralCameraPtr)
{
    std::string id(HyperspectralCameraIO<Derived>::ID);
    std::string obj(HyperspectralCameraIO<Derived>::OBJID);
    hdf5util::setAttribute(group, "IO", id);
    hdf5util::setAttribute(group, "CLASS", obj);

    // saving estimated and registrated pose
    m_matrixIO->save(group, "extrinsics", hyperspectralCameraPtr->extrinsics);
    m_matrixIO->save(group, "extrinsicsEstimate", hyperspectralCameraPtr->extrinsicsEstimate);

    std::vector<size_t> dim = {1, 1};
    std::vector<hsize_t> chunks = {1, 1};

    // saving focalLength
    doubleArr focalLength(new double[1]);
    focalLength[0] = hyperspectralCameraPtr->focalLength;
    m_arrayIO->save(group, "focalLength", dim, chunks, focalLength);

    // saving offsetAngle
    doubleArr offsetAngle(new double[1]);
    offsetAngle[0] = hyperspectralCameraPtr->offsetAngle;
    m_arrayIO->save(group, "offsetAngle", dim, chunks, offsetAngle);

    dim = {3, 1};
    chunks = {3, 1};

    doubleArr principal(new double[3]);
    principal[0] = hyperspectralCameraPtr->principal[0];
    principal[1] = hyperspectralCameraPtr->principal[1];
    principal[2] = hyperspectralCameraPtr->principal[2];
    m_arrayIO->save(group, "principal", dim, chunks, principal);

    doubleArr distortion(new double[3]);
    distortion[0] = hyperspectralCameraPtr->distortion[0];
    distortion[1] = hyperspectralCameraPtr->distortion[1];
    distortion[2] = hyperspectralCameraPtr->distortion[2];
    m_arrayIO->save(group, "distortion", dim, chunks, distortion);

    for (int i = 0; i < hyperspectralCameraPtr->panoramas.size(); i++)
    {
        HyperspectralPanoramaPtr panoramaPtr = hyperspectralCameraPtr->panoramas[i];

        ucharArr data(new unsigned char[hyperspectralCameraPtr->panoramas[i]->channels.size() *
                                        panoramaPtr->channels[0]->channel.rows *
                                        panoramaPtr->channels[0]->channel.cols]);

        std::memcpy(data.get(),
                    panoramaPtr->channels[0]->channel.data,
                    panoramaPtr->channels[0]->channel.rows *
                        panoramaPtr->channels[0]->channel.cols * sizeof(unsigned char));

        std::vector<size_t> dim = {hyperspectralCameraPtr->panoramas[i]->channels.size(),
                                   static_cast<size_t>(panoramaPtr->channels[0]->channel.rows),
                                   static_cast<size_t>(panoramaPtr->channels[0]->channel.cols)};

        for (int j = 1; j < hyperspectralCameraPtr->panoramas[i]->channels.size(); j++)
        {
            std::memcpy(data.get() + j * panoramaPtr->channels[j]->channel.rows *
                                         panoramaPtr->channels[j]->channel.cols,
                        panoramaPtr->channels[j]->channel.data,
                        panoramaPtr->channels[j]->channel.rows *
                            panoramaPtr->channels[j]->channel.cols * sizeof(unsigned char));
        }

        std::vector<hsize_t> chunks = {50, 50, 50};

        // generate group of panorama
        char buffer[sizeof(int) * 5];
        sprintf(buffer, "%08d", i);
        string nr_str(buffer);
        HighFive::Group panoramaGroup = hdf5util::getGroup(group, nr_str);

        // save panorama
        m_arrayIO->save(panoramaGroup, "frames", dim, chunks, data);

        // save timestamps
        doubleArr timestamps(new double[hyperspectralCameraPtr->panoramas[i]->channels.size()]);
        size_t pos = 0;
        for (auto channel : hyperspectralCameraPtr->panoramas[i]->channels)
        {
            timestamps[pos++] = channel->timestamp;
        }
        dim = {pos, 1, 1};
        chunks = {pos, 1, 1};
        m_arrayIO->save(panoramaGroup, "timestamps", dim, chunks, timestamps);
    }
}

template <typename Derived>
HyperspectralCameraPtr HyperspectralCameraIO<Derived>::load(uint scanPos)
{
    HyperspectralCameraPtr ret;

    char buffer[sizeof(int) * 5];
    sprintf(buffer, "%08d", scanPos);
    string nr_str(buffer);
    std::string basePath = "raw/" + nr_str + "/spectral/data";

    if (hdf5util::exist(m_file_access->m_hdf5_file, basePath))
    {
        HighFive::Group group = hdf5util::getGroup(m_file_access->m_hdf5_file, basePath);
        ret = load(group);
    }

    return ret;
}

template <typename Derived>
HyperspectralCameraPtr HyperspectralCameraIO<Derived>::loadHyperspectralCamera(uint scanPos)
{
    HyperspectralCameraPtr ret;

    char buffer[sizeof(int) * 5];
    sprintf(buffer, "%08d", scanPos);
    string nr_str(buffer);
    std::string basePath = "raw/" + nr_str + "/spectral/data";

    if (hdf5util::exist(m_file_access->m_hdf5_file, basePath))
    {
        HighFive::Group group = hdf5util::getGroup(m_file_access->m_hdf5_file, basePath);
        ret = load(group);
    }

    return ret;
}

template <typename Derived>
HyperspectralCameraPtr HyperspectralCameraIO<Derived>::load(HighFive::Group& group)
{
    HyperspectralCameraPtr ret(new HyperspectralCamera);

    if (!isHyperspectralCamera(group))
    {
        std::cout << "[Hdf5IO - HyperspectralCameraIO] WARNING: flags of " << group.getId()
                  << " are not correct." << std::endl;
        return ret;
    }

    // read extrinsics
    boost::optional<lvr2::Extrinsicsd> extrinsics =
        m_matrixIO->template load<lvr2::Extrinsicsd>(group, "extrinsics");
    if (extrinsics)
    {
        ret->extrinsics = extrinsics.get();
    }

    // read extrinsicsEstimate
    boost::optional<lvr2::Extrinsicsd> extrinsicsEstimate =
        m_matrixIO->template load<lvr2::Extrinsicsd>(group, "extrinsicsEstimate");
    if (extrinsicsEstimate)
    {
        ret->extrinsicsEstimate = extrinsicsEstimate.get();
    }

    // read focalLength
    if (group.exist("focalLength"))
    {
        std::vector<size_t> dimension;
        doubleArr focalLength = m_arrayIO->template load<double>(group, "focalLength", dimension);

        if (dimension.at(0) != 1)
        {
            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong focalLength dimension. The focalLength "
                         "will not be loaded."
                      << std::endl;
        }
        else
        {
            ret->focalLength = focalLength[0];
        }
    }

    // read offsetAngle
    if (group.exist("offsetAngle"))
    {
        std::vector<size_t> dimension;
        doubleArr offsetAngle = m_arrayIO->template load<double>(group, "offsetAngle", dimension);

        if (dimension.at(0) != 1)
        {
            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong offsetAngle dimension. The offsetAngle "
                         "will not be loaded."
                      << std::endl;
        }
        else
        {
            ret->offsetAngle = offsetAngle[0];
        }
    }

    // read principal
    if (group.exist("principal"))
    {
        std::vector<size_t> dimension;
        doubleArr principal = m_arrayIO->template load<double>(group, "principal", dimension);

        if (dimension.at(0) != 3)
        {
            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong principal dimension. The principal "
                         "will not be loaded."
                      << std::endl;
        }
        else
        {
            Vector3d p = {principal[0], principal[1], principal[2]};
            ret->principal = p;
        }
    }

    // read distortion
    if (group.exist("distortion"))
    {
        std::vector<size_t> dimension;
        doubleArr distortion = m_arrayIO->template load<double>(group, "distortion", dimension);

        if (dimension.at(0) != 3)
        {
            std::cout << "[Hdf5IO - ScanIO] WARNING: Wrong distortion dimension. The distortion "
                         "will not be loaded."
                      << std::endl;
        }
        else
        {
            Vector3d d = {distortion[0], distortion[1], distortion[2]};
            ret->distortion = d;
        }
    }

    // iterate over all panoramas
    for (std::string groupname : group.listObjectNames())
    {
        // load all scanCameras
        if (std::regex_match(groupname, std::regex("\\d{8}")))
        {
            HighFive::Group g = hdf5util::getGroup(group, "/" + groupname);

            std::vector<size_t> dim;
            ucharArr data = m_arrayIO->template load<uchar>(g, "frames", dim);

            std::vector<size_t> timeDim;
            doubleArr timestamps = m_arrayIO->template load<double>(g, "timestamps", timeDim);

            HyperspectralPanoramaPtr panoramaPtr(new HyperspectralPanorama);
            for (int i = 0; i < dim[0]; i++)
            {
                // img size ist dim[1] * dim[2]

                cv::Mat img = cv::Mat(dim[1], dim[2], CV_8UC1);
                std::memcpy(
                    img.data, data.get() + i * dim[1] * dim[2], dim[1] * dim[2] * sizeof(uchar));

                HyperspectralPanoramaChannelPtr channelPtr(new HyperspectralPanoramaChannel);
                channelPtr->channel = img;
                channelPtr->timestamp = timestamps[i];
                panoramaPtr->channels.push_back(channelPtr);
            }
            ret->panoramas.push_back(panoramaPtr);
        }
    }

    return ret;
}

template <typename Derived>
bool HyperspectralCameraIO<Derived>::isHyperspectralCamera(HighFive::Group& group)
{
    std::string id(HyperspectralCameraIO<Derived>::ID);
    std::string obj(HyperspectralCameraIO<Derived>::OBJID);
    return hdf5util::checkAttribute(group, "IO", id) &&
           hdf5util::checkAttribute(group, "CLASS", obj);
}

} // namespace hdf5features

} // namespace lvr2
