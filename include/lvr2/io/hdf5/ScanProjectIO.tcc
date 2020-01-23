namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
void ScanProjectIO<Derived>::save(const ScanProjectPtr& scanProjectPtr)
{
    int pos = 0;

    // iterate over all positions
    for (ScanPositionPtr scanPosPtr : (*scanProjectPtr).positions)
    {
        char buffer[sizeof(int) * 5];
        sprintf(buffer, "%05d", pos++);
        string nr_str(buffer);

        std::string basePath = "raw/" + nr_str + "/";

        std::cout << "writing scanPosition " << (pos - 1) << " to path " << basePath << std::endl;

        HighFive::Group scanPosGroup = hdf5util::getGroup(m_file_access->m_hdf5_file, basePath);

        m_scanPositionIO->template save(scanPosGroup, scanPosPtr);

        // for (ScanPtr(*scanPosPtr).scans)
        // {
        //     std::cout << "scan detected" << std::endl;
        //     // save scan to HDF5
        //     HighFive::Group scanGroup =
        //         hdf5util::getGroup(m_file_access->m_hdf5_file, basePath + "/scans/data");
        //     m_scanIO->template save(scanGroup, (*scanPosPtr).scan.get());
        // }
    }
}

template <typename Derived>
ScanProjectPtr ScanProjectIO<Derived>::load()
{
    ScanProjectPtr ret(new ScanProject());

    HighFive::Group hfscans = hdf5util::getGroup(m_file_access->m_hdf5_file, "/raw");
    size_t scanPoitions = hfscans.getNumberObjects();

    std::cout << "found " << scanPoitions << " possible scanPositions" << std::endl;

    // iterate over all possible scanPositions
    for (std::string groupname : hfscans.listObjectNames())
    {
        std::cout << groupname << std::endl;
        // TODO: Build group from groupname and check if it is a scanPosition
        // If it is a scanPosition, add it to the scanProject
    }

    return ret;
}

template <typename Derived>
ScanProjectPtr ScanProjectIO<Derived>::loadScanProject()
{
    return load();
}

} // namespace hdf5features

} // namespace lvr2
