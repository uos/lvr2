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

    // TODO: save camera model

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
        int pos = 0;
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

    // iterate over all panoramas
    for (std::string groupname : group.listObjectNames())
    {
        // load all scanCameras
        if (std::regex_match(groupname, std::regex("\\d{8}")))
        {
            HighFive::Group g = hdf5util::getGroup(group, "/" + groupname);

            std::vector<size_t> dim;
            ucharArr data = m_arrayIO->template load<uchar>(g, "frames", dim);

            HyperspectralPanoramaPtr panoramaPtr(new HyperspectralPanorama);
            for (int i = 0; i < dim[0]; i++)
            {
                // img size ist dim[1] * dim[2]

                cv::Mat img = cv::Mat(dim[1], dim[2], CV_8UC1);
                std::memcpy(
                    img.data, data.get() + i * dim[1] * dim[2], dim[1] * dim[2] * sizeof(uchar));

                HyperspectralPanoramaChannelPtr channelPtr(new HyperspectralPanoramaChannel);
                channelPtr->channel = img;
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
