namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
void ScanProjectIO<Derived>::save(const ScanProjectPtr& scanProjectPtr)
{
    int pos = 0;

    // iterate over all positions
    for (ScanPositionPtr scanPosPtr : scanProjectPtr->positions)
    {
        char buffer[sizeof(int) * 5];
        sprintf(buffer, "%08d", pos++);
        string nr_str(buffer);

        std::string basePath = "raw/" + nr_str;

        HighFive::Group scanPosGroup = hdf5util::getGroup(m_file_access->m_hdf5_file, basePath);

        m_scanPositionIO->save(scanPosGroup, scanPosPtr);
    }
}

template <typename Derived>
ScanProjectPtr ScanProjectIO<Derived>::load()
{
    ScanProjectPtr ret(new ScanProject());

    // iterate over all possible scanPositions
    if (hdf5util::exist(m_file_access->m_hdf5_file, "raw"))
    {
        HighFive::Group hfscans = hdf5util::getGroup(m_file_access->m_hdf5_file, "/raw");
        for (std::string groupname : hfscans.listObjectNames())
        {
            if (std::regex_match(groupname, std::regex("\\d{8}")))
            {
                HighFive::Group scanPosGroup = hdf5util::getGroup(hfscans, groupname, false);
                ScanPositionPtr scanPosition = m_scanPositionIO->load(scanPosGroup);
                ret->positions.push_back(scanPosition);
            }
        }
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
