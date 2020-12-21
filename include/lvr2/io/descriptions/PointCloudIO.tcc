namespace lvr2 
{

template<typename FeatureBase>
void PointCloudIO<FeatureBase>::savePointCloud(
    const std::string& groupName, 
    const PointBufferPtr& buffer)
{
    for(auto elem : *buffer)
    {
        m_vchannel_io->save(groupName, elem.first, elem.second);
    }
}

template<typename FeatureBase>
void PointCloudIO<FeatureBase>::savePointCloud(
    const std::string& groupName, 
    const std::string& container, 
    const PointBufferPtr& buffer)
{
    boost::filesystem::path p(container);
    if(p.extension() == "") {
        // no extension: assuming to store each channel
        savePointCloud(groupName + "/" + container, buffer );
    } else {
        m_featureBase->m_kernel->savePointBuffer(groupName, container, buffer);
    }
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
    PointBufferPtr ret;

    using VChannelT = typename PointBuffer::val_type;

    // load all channel in group
    for(auto meta : m_featureBase->m_kernel->metas(group) ) 
    {
        if(meta.second["sensor_type"])
        {
            if(meta.second["sensor_type"].template as<std::string>() == "Channel")
            {   
                boost::optional<VChannelT> copt = m_vchannel_io->template loadVariantChannel<VChannelT>(group, meta.first);
                
                if(copt)
                {
                    if(!ret)
                    {
                        ret.reset(new PointBuffer);
                    }
                    (*ret)[meta.first] = *copt;
                }
            }
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