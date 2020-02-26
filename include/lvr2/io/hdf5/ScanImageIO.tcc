namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
void ScanImageIO<Derived>::save(uint scanPos,
                                uint camNr,
                                uint imgNr,
                                const ScanImagePtr& scanImagePtr)
{
    // TODO
}

template <typename Derived>
void ScanImageIO<Derived>::save(HighFive::Group& group,
                                uint camNr,
                                uint imgNr,
                                const ScanImagePtr& scanImagePtr)
{
    // TODO: check wether the given group is type ScanPositionIO

    // TODO
}

template <typename Derived>
void ScanImageIO<Derived>::save(HighFive::Group& group,
                                uint imgNr,
                                const ScanImagePtr& scanImagePtr)
{
    // TODO: check wether the given group is type ScanCameraIO

    std::string id(ScanImageIO<Derived>::ID);
    std::string obj(ScanImageIO<Derived>::OBJID);
    hdf5util::setAttribute(group, "IO", id);
    hdf5util::setAttribute(group, "CLASS", obj);

    char buffer[sizeof(int) * 5];
    sprintf(buffer, "%08d", imgNr);
    string nr_str(buffer);

    HighFive::Group imgPosGroup = hdf5util::getGroup(group, nr_str);
    save(imgPosGroup, scanImagePtr);
}

template <typename Derived>
void ScanImageIO<Derived>::save(HighFive::Group& group, const ScanImagePtr& scanImagePtr)
{
    std::string id(ScanImageIO<Derived>::ID);
    std::string obj(ScanImageIO<Derived>::OBJID);
    hdf5util::setAttribute(group, "IO", id);
    hdf5util::setAttribute(group, "CLASS", obj);

    // save image with imageIO
    m_imageIO->save(group, "image", scanImagePtr->image);

    // save extrinsics
    m_matrixIO->save(group, "extrinsics", scanImagePtr->extrinsics);
    m_matrixIO->save(group, "extrinsicsEstimate", scanImagePtr->extrinsicsEstimate);
}

template <typename Derived>
ScanImagePtr ScanImageIO<Derived>::load(uint scanPos, uint camNr, uint imgNr)
{
    ScanImagePtr ret;

    // TODO

    return ret;
}

template <typename Derived>
ScanImagePtr ScanImageIO<Derived>::load(HighFive::Group& group, uint camNr, uint imgNr)
{
    ScanImagePtr ret;

    // check wether the given group is type ScanProjectIO

    // TODO

    return ret;
}

template <typename Derived>
ScanImagePtr ScanImageIO<Derived>::load(HighFive::Group& group, uint imgNr)
{
    ScanImagePtr ret;

    // check wether the given group is type ScanPositionIO

    // TODO

    return ret;
}

template <typename Derived>
ScanImagePtr ScanImageIO<Derived>::load(HighFive::Group& group)
{
    ScanImagePtr ret(new ScanImage());

    // load image with imageIO
    boost::optional<cv::Mat> image = m_imageIO->load(group, "image");
    if (image)
    {
        ret->image = image.get();
    }

    // load extrinsics
    boost::optional<lvr2::Extrinsicsd> extrinsics =
        m_matrixIO->template load<lvr2::Extrinsicsd>(group, "extrinsics");
    if (extrinsics)
    {
        ret->extrinsics = extrinsics.get();
    }

    boost::optional<lvr2::Extrinsicsd> extrinsicsEstimate =
        m_matrixIO->template load<lvr2::Extrinsicsd>(group, "extrinsicsEstimate");
    if (extrinsicsEstimate)
    {
        ret->extrinsicsEstimate = extrinsicsEstimate.get();
    }

    return ret;
}

template <typename Derived>
bool ScanImageIO<Derived>::isScanImage(HighFive::Group& group)
{
    std::string id(ScanImageIO<Derived>::ID);
    std::string obj(ScanImageIO<Derived>::OBJID);
    return hdf5util::checkAttribute(group, "IO", id) &&
           hdf5util::checkAttribute(group, "CLASS", obj);
}

} // namespace hdf5features

} // namespace lvr2
