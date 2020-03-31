namespace lvr2 
{

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
bool PointCloudIO<FeatureBase>::isPointCloud(
    HighFive::Group& group)
{
    return true;
}

} // namespace lvr2 