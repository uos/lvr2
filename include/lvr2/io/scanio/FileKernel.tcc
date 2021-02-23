namespace lvr2
{


template<typename T>
boost::shared_array<T> FileKernel::loadArray(
    const std::string& group, 
    const std::string& container, 
    std::vector<size_t>& dims) const
{
    if constexpr (std::is_same_v<T, char>)
    {
        return loadCharArray(group, container, dims);
    } else 
    if constexpr (std::is_same_v<T, unsigned char>)
    {
        return loadUCharArray(group, container, dims);
    } else 
    if constexpr (std::is_same_v<T, short>)
    {
        return loadShortArray(group, container, dims);
    } else 
    if constexpr (std::is_same_v<T, unsigned short>)
    {
        return loadUShortArray(group, container, dims);
    } else 
    if constexpr (std::is_same_v<T, uint16_t>)
    {
        return loadUInt16Array(group, container, dims);
    } else 
    if constexpr (std::is_same_v<T, int>)
    {
        return loadIntArray(group, container, dims);
    } else 
    if constexpr (std::is_same_v<T, unsigned int>)
    {
        return loadUIntArray(group, container, dims);
    } else 
    if constexpr (std::is_same_v<T, long int>)
    {
        return loadLIntArray(group, container, dims);
    } else 
    if constexpr (std::is_same_v<T, unsigned long int>)
    {
        return loadULIntArray(group, container, dims);
    } else 
    if constexpr (std::is_same_v<T, float>)
    {
        return loadFloatArray(group, container, dims);
    } else 
    if constexpr (std::is_same_v<T, double>)
    {
        return loadDoubleArray(group, container, dims);
    } else 
    if constexpr (std::is_same_v<T, bool>)
    {
        return loadBoolArray(group, container, dims);
    }
    else {
        // boost::type_info<>
        std::cout << "[FileKernel] WARNING: not implemented type " << boost::typeindex::type_id<T>().pretty_name() << std::endl;
        throw std::runtime_error("loadArray fail");
    }
}

template<typename T>
void FileKernel::saveArray(
    const std::string& groupName, 
    const std::string& datasetName, 
    const std::vector<size_t>& dimensions, 
    const boost::shared_array<T>& data) const
{
    if constexpr (std::is_same_v<T, char>)
    {
        return saveCharArray(groupName, datasetName, dimensions, data);
    } else 
    if constexpr (std::is_same_v<T, unsigned char>)
    {
        return saveUCharArray(groupName, datasetName, dimensions, data);
    } else 
    if constexpr (std::is_same_v<T, short>)
    {
        return saveShortArray(groupName, datasetName, dimensions, data);
    } else 
    if constexpr (std::is_same_v<T, unsigned short>)
    {
        return saveUShortArray(groupName, datasetName, dimensions, data);
    } else 
    if constexpr (std::is_same_v<T, uint16_t>)
    {
        return saveUInt16Array(groupName, datasetName, dimensions, data);
    } else 
    if constexpr (std::is_same_v<T, int>)
    {
        return saveIntArray(groupName, datasetName, dimensions, data);
    } else 
    if constexpr (std::is_same_v<T, unsigned int>)
    {
        return saveUIntArray(groupName, datasetName, dimensions, data);
    } else 
    if constexpr (std::is_same_v<T, long int>)
    {
        return saveLIntArray(groupName, datasetName, dimensions, data);
    } else 
    if constexpr (std::is_same_v<T, unsigned long int>)
    {
        return saveULIntArray(groupName, datasetName, dimensions, data);
    } else 
    if constexpr (std::is_same_v<T, float>)
    {
        return saveFloatArray(groupName, datasetName, dimensions, data);
    } else 
    if constexpr (std::is_same_v<T, double>)
    {
        return saveDoubleArray(groupName, datasetName, dimensions, data);
    } else 
    if constexpr (std::is_same_v<T, bool>)
    {
        return saveBoolArray(groupName, datasetName, dimensions, data);
    }
    else {
        // boost::type_info<>
        std::cout << "[FileKernel] WARNING: not implemented type " << boost::typeindex::type_id<T>().pretty_name() << std::endl;
        throw std::runtime_error("saveArray fail");
    }
}


} // namespace lvr2