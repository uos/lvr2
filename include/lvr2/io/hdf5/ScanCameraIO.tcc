namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
void ScanCameraIO<Derived>::save(uint scanPos, uint camNr, const ScanCameraPtr& ScanCameraPtr)
{
    // TODO
}

template <typename Derived>
void ScanCameraIO<Derived>::save(HighFive::Group& group,
                                 uint camNr,
                                 const ScanCameraPtr& ScanCameraPtr)
{
    // check wether the given group is type ScanPositionIO

    // TODO
}

template <typename Derived>
void ScanCameraIO<Derived>::save(HighFive::Group& group, const ScanCameraPtr& ScanCameraPtr)
{
    std::string id(ScanCameraIO<Derived>::ID);
    std::string obj(ScanCameraIO<Derived>::OBJID);
    hdf5util::setAttribute(group, "IO", id);
    hdf5util::setAttribute(group, "CLASS", obj);

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
    ScanCameraPtr ret(new Scan());

    // TODO

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
