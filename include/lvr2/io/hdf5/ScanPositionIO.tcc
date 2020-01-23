namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
void ScanPositionIO<Derived>::save(std::string name, const ScanPositionPtr& scanPositionPtr)
{
    // TODO call save with group
}

template <typename Derived>
void ScanPositionIO<Derived>::save(HighFive::Group& group, const ScanPositionPtr& scanPositionPtr)
{
    int pos = 0;

    for (ScanPtr scanPtr : (*scanPositionPtr).scans)
    {
        char buffer[sizeof(int) * 5];
        sprintf(buffer, "%05d", pos++);
        string nr_str(buffer);

        std::string path = "/scans/data/" + nr_str;
        std::cout << path << std::endl;

        HighFive::Group scanGroup = hdf5util::getGroup(group, path);
        std::cout << "path" << std::endl;

        m_scanIO->template save(scanGroup, scanPtr);
    }
}

template <typename Derived>
ScanPositionPtr ScanPositionIO<Derived>::load(HighFive::Group& group)
{
    ScanPositionPtr ret;

    if (!isScanPosition(group))
    {
        std::cout << "[Hdf5IO - ScanPositionIO] WARNING: flags of " << group.getId()
                  << " are not correct." << std::endl;
        return ret;
    }

    // load all scans

    // HighFive::Group hfscans = hdf5util::getGroup(m_file_access->m_hdf5_file, "/raw");
    // size_t scans = hfscans.getNumberObjects();

    // std::cout << "found " << scans << " possible scanPositions" << std::endl;

    // // iterate over all possible scanPositions
    // for (std::string groupname : hfscans.listObjectNames())
    // {
    //     std::cout << groupname << std::endl;
    //     // TODO: Build group from groupname and check if it is a scanPosition
    //     // If it is a scanPosition, add it to the scanProject
    // }

    return ret;
}

template <typename Derived>
ScanPositionPtr ScanPositionIO<Derived>::load(std::string name)
{
    ScanPositionPtr ret;

    if (hdf5util::exist(m_file_access->m_hdf5_file, name))
    {
        HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, name, false);
        ret = load(g);
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
