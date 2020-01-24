namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
void ScanImageIO<Derived>::save(uint scanPos,
                                uint camNr,
                                uint imgNr,
                                const ScanImagePtr& ScanImagePtr)
{
    // TODO
}

template <typename Derived>
void ScanImageIO<Derived>::save(HighFive::Group& group,
                                uint camNr,
                                uint imgNr,
                                const ScanImagePtr& ScanImagePtr)
{
    // check wether the given group is type ScanPositionIO

    // TODO
}

template <typename Derived>
void ScanImageIO<Derived>::save(HighFive::Group& group,
                                uint imgNr,
                                const ScanImagePtr& ScanImagePtr)
{
    // TODO
}

template <typename Derived>
void ScanImageIO<Derived>::save(HighFive::Group& group, const ScanImagePtr& ScanImagePtr)
{
    std::string id(ScanImageIO<Derived>::ID);
    std::string obj(ScanImageIO<Derived>::OBJID);
    hdf5util::setAttribute(group, "IO", id);
    hdf5util::setAttribute(group, "CLASS", obj);

    // TODO
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
    ScanImagePtr ret(new Scan());

    // TODO

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
