namespace lvr2 
{

template<typename FeatureBase>
void PolygonIO<FeatureBase>::savePolygon(
    const std::string& groupName, 
    const std::string& container, 
    const PolygonPtr& buffer)
{
    m_featureBase->m_kernel->savePointBuffer(groupName, container, buffer);
}

template<typename FeatureBase>
PolygonPtr PolygonIO<FeatureBase>::loadPolygon(const std::string& group, const std::string& name)
{
    return m_featureBase->m_kernel->loadPointBuffer(group, name);
}


template<typename FeatureBase>
bool PointCloudIO<FeatureBase>::isPolygon(
    HighFive::Group& group)
{
    return true;
}

} // namespace lvr2 