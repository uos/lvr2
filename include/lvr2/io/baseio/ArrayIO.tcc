namespace lvr2 
{

namespace baseio
{

template<typename BaseIO>
ucharArr ArrayIO<BaseIO>::loadUCharArray(const std::string &group, const std::string &container, std::vector<size_t> &dims) const
{
    return m_BaseIO->m_kernel->loadUCharArray(group, container, dims);
}

template<typename BaseIO>
floatArr ArrayIO<BaseIO>::loadFloatArray(const std::string &group, const std::string &container, std::vector<size_t> &dims) const
{
    return m_BaseIO->m_kernel->loadFloatArray(group, container, dims);
}

template<typename BaseIO>
doubleArr ArrayIO<BaseIO>::loadDoubleArray(const std::string &group, const std::string &container, std::vector<size_t> &dims) const
{
    return m_BaseIO->m_kernel->loadDoubleArray(group, container, dims);
}

template<typename BaseIO>
intArr ArrayIO<BaseIO>::loadIntArray(const std::string &group, const std::string &container, std::vector<size_t> &dims) const
{
    return m_BaseIO->m_kernel->loadIntArray(group, container, dims);
}

template<typename BaseIO>
void ArrayIO<BaseIO>::saveFloatArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t> &dimensions, const boost::shared_array<float>& data) const
{
    m_BaseIO->m_kernel->saveFloatArray(groupName, datasetName, dimensions, data);
}

template<typename BaseIO>
void ArrayIO<BaseIO>::saveDoubleArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t> &dimensions, const boost::shared_array<double>& data) const
{
    m_BaseIO->m_kernel->saveDoubleArray(groupName, datasetName, dimensions, data);
}

template<typename BaseIO>
void ArrayIO<BaseIO>::saveUCharArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t> &dimensions, const boost::shared_array<unsigned char>& data) const
{
    m_BaseIO->m_kernel->saveUCharArray(groupName, datasetName, dimensions, data);
}
template<typename BaseIO>
void ArrayIO<BaseIO>::saveIntArray(const std::string& groupName, const std::string& datasetName, const std::vector<size_t> &dimensions, const boost::shared_array<int>& data) const
{
    m_BaseIO->m_kernel->saveIntArray(groupName, datasetName, dimensions, data);
}

} // namespace scanio

} // namespace lvr2