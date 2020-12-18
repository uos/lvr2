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
    m_featureBase->m_kernel->savePointBuffer(groupName, container, buffer);
}

template<typename FeatureBase>
PointBufferPtr PointCloudIO<FeatureBase>::loadPointCloud(const std::string& group, const std::string& name)
{
    return m_featureBase->m_kernel->loadPointBuffer(group, name);
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
bool PointCloudIO<FeatureBase>::isPointCloud(
    const std::string& group, const std::string& name)
{
    // TODO: better not read anything for isPointCloud check
    return static_cast<bool>(loadPointCloud(group, name));
}

} // namespace lvr2 