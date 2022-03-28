namespace lvr2 
{

template<typename BaseIO>
void PolygonIO<BaseIO>::savePolygon(
    const std::string& groupName, 
    const std::string& container, 
    const PolygonPtr& buffer)
{
    m_baseIO->m_kernel->savePointBuffer(groupName, container, buffer);
}

template<typename BaseIO>
PolygonPtr PolygonIO<BaseIO>::loadPolygon(const std::string& group, const std::string& name)
{
    return m_baseIO->m_kernel->loadPointBuffer(group, name);
}


template<typename BaseIO>
bool PointCloudIO<BaseIO>::isPolygon(
    HighFive::Group& group)
{
    return true;
}

} // namespace lvr2 