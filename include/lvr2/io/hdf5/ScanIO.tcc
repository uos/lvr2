namespace lvr2
{

namespace hdf5features
{

template <typename Derived>
void ScanIO<Derived>::save(std::string name, const ScanPtr& buffer)
{
    HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, name, true);

    save(g, buffer);
}
template <typename Derived>
void MeshIO<Derived>::save(HighFive::Group& group, const MeshBufferPtr& buffer)
{
    // TODO
}


template <typename Derived>
ScanPtr ScanIO<Derived>::load(std::string name)
{
    ScanPtr ret;

    if (hdf5util::exist(m_file_access->m_hdf5_file, name))
    {
        HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, name, false);
        ret               = load(g);
    }

    return ret;
}

template <typename Derived>
ScanPtr ScanIO<Derived>::loadScan(std::string name)
{
    m_mesh_name = name;
    return load(name);
}

template <typename Derived>
ScanPtr ScanIO<Derived>::load(HighFive::Group& group)
{
    // TODO
    Scan ret;
    return ret;
}

template <typename Derived>
bool MeshIO<Derived>::isScan(
        HighFive::Group& group)
{
    std::string id(ScanIO<Derived>::ID);
    std::string obj(ScanIO<Derived>::OBJID);
    return hdf5util::checkAttribute(group, "IO", id)
           && hdf5util::checkAttribute(group, "CLASS", obj);
}



} // namespace hdf5features

} // namespace lvr2
