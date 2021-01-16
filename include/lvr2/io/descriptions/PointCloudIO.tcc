namespace lvr2 
{

template<typename FeatureBase>
void PointCloudIO<FeatureBase>::save(
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
        m_featureBase->m_kernel->savePointBuffer(group, name, pcl);
    }
}

template<typename FeatureBase>
void PointCloudIO<FeatureBase>::save(
    const std::string& groupandname, 
    PointBufferPtr pcl) const
{
    for(auto elem : *pcl)
    {
        std::cout << "-- save " << elem.first << std::endl;
        m_vchannel_io->save(groupandname, elem.first, elem.second);
    }
}

template<typename FeatureBase>
void PointCloudIO<FeatureBase>::savePointCloud(
    const std::string& group, 
    const std::string& name, 
    PointBufferPtr pcl) const
{
    save(group, name, pcl);
}

template<typename FeatureBase>
void PointCloudIO<FeatureBase>::savePointCloud(
    const std::string& groupandname,
    PointBufferPtr pcl) const
{
    save(groupandname, pcl);
}

template<typename FeatureBase>
PointBufferPtr PointCloudIO<FeatureBase>::loadPointCloud(
    const std::string& group, 
    const std::string& name)
{
    boost::filesystem::path p(name);
    if(p.extension() == "") {
        // no extension: assuming to store each channel
        return loadPointCloud(group + "/" + name);
    } else {
        return m_featureBase->m_kernel->loadPointBuffer(group, name);
    }
}

template<typename FeatureBase>
PointBufferPtr PointCloudIO<FeatureBase>::loadPointCloud( 
    const std::string& group,
    const std::string& container, 
    ReductionAlgorithmPtr reduction)
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

template<typename FeatureBase>
PointBufferPtr PointCloudIO<FeatureBase>::loadPointCloud(
    const std::string& group)
{
    std::cout << "[IO: PointCloudIO - load]: " << group << std::endl;
    PointBufferPtr ret;

    using VChannelT = typename PointBuffer::val_type;

    // load all channel in group
    for(auto meta : m_featureBase->m_kernel->metas(group, "Channel") )
    {
        boost::optional<VChannelT> copt = m_vchannel_io->template loadVariantChannel<VChannelT>(group, meta.first);
        
        if(copt)
        {
            if(!ret)
            {
                ret.reset(new PointBuffer);
            }
            // add channel
            (*ret)[meta.first] = *copt;
        }
    }

    return ret;
}

template<typename FeatureBase>
bool PointCloudIO<FeatureBase>::isPointCloud(
    const std::string& group, const std::string& name)
{
    // TODO: better not read anything for isPointCloud check
    return static_cast<bool>(loadPointCloud(group, name));
}

} // namespace lvr2 