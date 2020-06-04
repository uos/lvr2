namespace lvr2 
{

template<typename FeatureBase>
ucharArr ArrayIO<FeatureBase>::loadUCharArray(const std::string &group, const std::string &container, std::vector<size_t> &dims) const
{
    return m_featureBase->m_kernel->loadUCharArray(group, container, dims);
}

template<typename FeatureBase>
floatArr ArrayIO<FeatureBase>::loadFloatArray(const std::string &group, const std::string &container, std::vector<size_t> &dims) const
{
    return m_featureBase->m_kernel->loadFloatArray(group, container, dims);
}

template<typename FeatureBase>
doubleArr ArrayIO<FeatureBase>::loadDoubleArray(const std::string &group, const std::string &container, std::vector<size_t> &dims) const
{
    return m_featureBase->m_kernel->loadDoubleArray(group, container, dims);
}

template<typename FeatureBase>
void ArrayIO<FeatureBase>::saveFloatArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t> &dimensions, const boost::shared_array<float>& data) const
{
    m_featureBase->m_kernel->saveFloatArray(groupName, datasetName, dimensions, data);
}

template<typename FeatureBase>
void ArrayIO<FeatureBase>::saveDoubleArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t> &dimensions, const boost::shared_array<double>& data) const
{
    m_featureBase->m_kernel->saveDoubleArray(groupName, datasetName, dimensions, data);
}

template<typename FeatureBase>
void ArrayIO<FeatureBase>::saveUCharArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t> &dimensions, const boost::shared_array<unsigned char>& data) const
{
    m_featureBase->m_kernel->saveUCharArray(groupName, datasetName, dimensions, data);
}

} // namespace lvr2