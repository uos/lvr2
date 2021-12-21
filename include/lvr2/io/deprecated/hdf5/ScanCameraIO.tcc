namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
void ScanCameraIO<Derived>::save(uint scanPos, uint camNr, const ScanCameraPtr& scanCameraPtr)
{
    // TODO
}

template <typename Derived>
void ScanCameraIO<Derived>::save(HighFive::Group& group,
                                 uint camNr,
                                 const ScanCameraPtr& scanCameraPtr)
{
    // check wether the given group is type ScanPositionIO

    // TODO
}

template <typename Derived>
void ScanCameraIO<Derived>::save(HighFive::Group& group, const ScanCameraPtr& scanCameraPtr)
{
    std::string id(ScanCameraIO<Derived>::ID);
    std::string obj(ScanCameraIO<Derived>::OBJID);
    hdf5util::setAttribute(group, "IO", id);
    hdf5util::setAttribute(group, "CLASS", obj);

    uint pos = 0;
    for (ScanImagePtr scanImagePtr : scanCameraPtr->images)
    {
        m_scanImageIO->save(group, pos++, scanImagePtr);
        // TODO
    }

    // TODO
}

template <typename Derived>
ScanCameraPtr ScanCameraIO<Derived>::load(uint scanPos, uint camNr)
{
    ScanCameraPtr ret;

    // TODO

    return ret;
}

template <typename Derived>
ScanCameraPtr ScanCameraIO<Derived>::load(HighFive::Group& group, uint camNr)
{
    ScanCameraPtr ret;

    // check wether the given group is type ScanPositionIO

    // TODO

    return ret;
}

template <typename Derived>
ScanCameraPtr ScanCameraIO<Derived>::load(HighFive::Group& group)
{
    ScanCameraPtr ret(new ScanCamera());

    // TODO

    for (std::string groupname : group.listObjectNames())
    {
        if (std::regex_match(groupname, std::regex("\\d{8}")))
        {
            HighFive::Group g = hdf5util::getGroup(group, "/" + groupname);
            ScanImagePtr scanImagePtr = m_scanImageIO->load(g);
            ret->images.push_back(scanImagePtr);
        }
    }

    return ret;
}

template <typename Derived>
bool ScanCameraIO<Derived>::isScanCamera(HighFive::Group& group)
{
    std::string id(ScanCameraIO<Derived>::ID);
    std::string obj(ScanCameraIO<Derived>::OBJID);
    return hdf5util::checkAttribute(group, "IO", id) &&
           hdf5util::checkAttribute(group, "CLASS", obj);
}

} // namespace hdf5features

} // namespace lvr2
