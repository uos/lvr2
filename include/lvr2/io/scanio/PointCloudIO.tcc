namespace lvr2 
{

namespace scanio
{

template<typename BaseIO>
void PointCloudIO<BaseIO>::save(
    const std::string& group, 
    const std::string& name,
    PointBufferPtr pcl) const
{
    boost::filesystem::path p(name);
    if(p.extension() == "")
    {
        std::string groupandname = group + "/" + name;
        save(groupandname, pcl);
    } else {
        m_baseIO->m_kernel->savePointBuffer(group, name, pcl);
    }
}

template<typename BaseIO>
void PointCloudIO<BaseIO>::save(
    const std::string& groupandname, 
    PointBufferPtr pcl) const
{
    lvr2::logout::get() << lvr2::info << "Storing each channel individually" << lvr2::endl;
    for(auto elem : *pcl)
    {
        m_vchannel_io->save(groupandname, elem.first, elem.second);
    }
}

template<typename BaseIO>
PointBufferPtr PointCloudIO<BaseIO>::load(
    const std::string& group, 
    const std::string& name) const
{
    // lvr2::logout::get() << "[IO: PointCloudIO - load]: " << group << ", " << name << lvr2::endl;
    boost::filesystem::path p(name);
    if(p.extension() == "") {
        // no extension: assuming to store each channel
        return loadPointCloud(group + "/" + name);
    } else {
        return m_baseIO->m_kernel->loadPointBuffer(group, name);
    }
}

template<typename BaseIO>
PointBufferPtr PointCloudIO<BaseIO>::load(
    const std::string& group) const
{
    // lvr2::logout::get() << "[IO: PointCloudIO - load]: " << group << lvr2::endl;
    PointBufferPtr ret;

    using VChannelT = typename PointBuffer::val_type;

    // load all channel in group
    for(auto meta : m_baseIO->m_kernel->metas(group, "channel") )
    {
        boost::optional<VChannelT> copt = m_vchannel_io->template loadVariantChannel<VChannelT>(group, meta.first);
        if(copt)
        {
            if(!ret)
            {
                ret = std::make_shared<PointBuffer>();
            }
            // add channel
            (*ret)[meta.first] = *copt;
        }
    }

    return ret;
}

template<typename BaseIO>
PointBufferPtr PointCloudIO<BaseIO>::load( 
    const std::string& group,
    const std::string& container, 
    ReductionAlgorithmPtr reduction) const
{
    if(reduction)
    {
        PointBufferPtr buffer = loadPointCloud(group, container);
        reduction->setPointBuffer(buffer);
        return reduction->getReducedPoints();
    } else {
        return loadPointCloud(group, container);
    }
}

template<typename BaseIO>
void PointCloudIO<BaseIO>::savePointCloud(
    const std::string& group, 
    const std::string& name, 
    PointBufferPtr pcl) const
{
    save(group, name, pcl);
}

template<typename BaseIO>
void PointCloudIO<BaseIO>::savePointCloud(
    const std::string& groupandname,
    PointBufferPtr pcl) const
{
    save(groupandname, pcl);
}

template<typename BaseIO>
PointBufferPtr PointCloudIO<BaseIO>::loadPointCloud(
    const std::string& group, 
    const std::string& name) const
{
    return load(group, name);
}

template<typename BaseIO>
PointBufferPtr PointCloudIO<BaseIO>::loadPointCloud(
    const std::string& group) const
{
    return load(group);
}

template<typename BaseIO>
PointBufferPtr PointCloudIO<BaseIO>::loadPointCloud( 
    const std::string& group,
    const std::string& container, 
    ReductionAlgorithmPtr reduction) const
{
    return load(group, container, reduction);
}

} // namespace scanio

} // namespace lvr2 