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

        ucharArr data(
            new unsigned char[hyperspectralCameraPtr->panoramas[i]->channels.size() *
                              panoramaPtr->channels[0].rows * panoramaPtr->channels[0].cols]);

        std::memcpy(data.get(),
                    panoramaPtr->channels[0].data,
                    panoramaPtr->channels[0].rows * panoramaPtr->channels[0].cols *
                        sizeof(unsigned char));

        std::vector<size_t> dim = {hyperspectralCameraPtr->panoramas[i]->channels.size(),
                                   static_cast<size_t>(panoramaPtr->channels[0].rows),
                                   static_cast<size_t>(panoramaPtr->channels[0].cols)};

        for (int j = 1; j < hyperspectralCameraPtr->panoramas[i]->channels.size(); j++)
        {
            std::memcpy(data.get() +
                            j * panoramaPtr->channels[0].rows * panoramaPtr->channels[0].cols,
                        panoramaPtr->channels[0].data,
                        panoramaPtr->channels[0].rows * panoramaPtr->channels[0].cols *
                            sizeof(unsigned char));
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
        for (double t : hyperspectralCameraPtr->panoramas[i]->timestamps)
        {
            timestamps[pos++] = t;
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

    std::cout << "Hyperspectral found" << std::endl;

    // HyperspectralImagePtr imagePtr = hyperspectralCameraPtr->panoramas[0];

    // ucharArr data(new unsigned char[hyperspectralCameraPtr->panoramas.size() *
    //                                 imagePtr->panorama.rows * imagePtr->panorama.cols]);

    // std::memcpy(data.get(),
    //             imagePtr->panorama.data,
    //             imagePtr->panorama.rows * imagePtr->panorama.cols * sizeof(unsigned char));

    // std::vector<size_t> dim = {hyperspectralCameraPtr->panoramas.size(),
    //                            static_cast<size_t>(imagePtr->panorama.rows),
    //                            static_cast<size_t>(imagePtr->panorama.cols)};

    // for (int i = 1; i < hyperspectralCameraPtr->panoramas.size(); i++)
    // {
    //     std::cout << i << std::endl;
    //     imagePtr = hyperspectralCameraPtr->panoramas[i];

    //     std::memcpy(data.get() + i * (imagePtr->panorama.rows * imagePtr->panorama.cols),
    //                 imagePtr->panorama.data,
    //                 imagePtr->panorama.rows * imagePtr->panorama.cols * sizeof(unsigned char));
    // }

    // std::vector<hsize_t> chunks = {50, 50, 50};

    // std::vector<size_t> dim;
    // ucharArr data = m_arrayIO->template load<uchar>(group, "frames", dim);

    // for (int i = 0; i < dim[0]; i++)
    // {
    //     // img size ist dim[1] * dim[2]

    //     cv::Mat img = cv::Mat(dim[1], dim[2], CV_8UC1);
    //     std::memcpy(img.data, data.get() + i * dim[1] * dim[2], dim[1] * dim[2] * sizeof(uchar));

    //     HyperspectralPanoramaPtr panoramaPtr(new HyperspectralPanorama);
    //     panoramaPtr->channels.push_back(img);
    //     ret->panoramas.push_back(panoramaPtr);
    // }

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
