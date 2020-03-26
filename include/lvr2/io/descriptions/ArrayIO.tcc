namespace lvr2 
{

template<typename FeatureBase>
template<typename T>
boost::shared_array<T> ArrayIO<FeatureBase>::load(
    std::string groupName,
    std::string datasetName,
    size_t& size)
{
    return m_featureBase->m_kernel->template loadArray<T>(groupName, datasetName, size);
}

template<typename FeatureBase>
template<typename T>
boost::shared_array<T> ArrayIO<FeatureBase>::load(
    std::string groupName,
    std::string datasetName,
    std::vector<size_t>& dim)
{
   return m_featureBase->m_kernel->template loadArray(groupName, datasetName, dim);
}


template<typename FeatureBase>
template<typename T>
void ArrayIO<FeatureBase>::save(
    std::string groupName,
    std::string datasetName,
    size_t size,
    boost::shared_array<T> data)
{
    m_featureBase->m_kernel->template saveArray(groupName, datasetName, data, size);
}

template<typename FeatureBase>
template<typename T>
void ArrayIO<FeatureBase>::save(
        std::string groupName,
        std::string datasetName,
        std::vector<size_t>& dimensions,
        boost::shared_array<T> data)
{
   m_featureBase->m_kernel->template saveArray(groupName, datasetName, data, dimensions);
}


} // namespace lvr2