namespace lvr2 
{

template<typename FeatureBase>
template<typename T> 
ChannelOptional<T> ChannelIO<FeatureBase>::load(
    std::string group, std::string name)
{   
    ChannelOptional<T> ret;
    Channel<T> c;
    
    if(load(group, name, c))
    {
        ret = c;
    }

    return ret;
}

template<typename FeatureBase>
FloatChannelOptional ChannelIO<FeatureBase>::loadFloatChannel(
    std::string groupName, std::string datasetName)
{
    return load<float>(groupName, datasetName);
}

template<typename FeatureBase>
template<typename T>
void ChannelIO<FeatureBase>::save(
    std::string group,
    std::string name,
    const Channel<T>& channel) const
{
    if constexpr(std::is_same<T, float>::value ) {
        _save(group, name, channel);
    } else if constexpr(std::is_same<T, unsigned char>::value ) {
        _save(group, name, channel);
    } else if constexpr(std::is_same<T, double>::value ) {
        _save(group, name, channel);
    } else if constexpr(std::is_same<T, int>::value ) {
        _save(group, name, channel);
    } else if constexpr(std::is_same<T, uint16_t>::value ) {
        _save(group, name, channel);
    } else {
        // NOT IMPLEMENTED TYPE TO WRITE
        std::cout << "[ChannelIO] Type not implemented for " << group << "/" << name << std::endl;
    }
}

// PROTECTED

template<typename FeatureBase>
bool ChannelIO<FeatureBase>::load(  std::string group,
            std::string name,
            Channel<float>& channel)
{
    std::vector<size_t> dims;
    floatArr arr = m_featureBase->m_kernel->loadFloatArray(group, name, dims);

    // Check if load was successfull. Channels should always
    // have a dimensionality of [width x n]. So dim.size() 
    // has to be 2.
    if(arr != nullptr)
    {
        if(dims.size() != 2)
        {
            std::cout << timestamp << "ChannelIO.load(): Trying to load data with dim = " 
                      << dims.size() << "." << std::endl;
            std::cout << timestamp << "Should be 2." << std::endl;
            return false;
        }
        channel = Channel<float>(dims[1], dims[0], arr);
        return true;
    } 

    return false;
}

template<typename FeatureBase>
bool ChannelIO<FeatureBase>::load(  
    std::string group,
    std::string name,
    Channel<unsigned char>& channel)
{
    std::vector<size_t> dims;
    ucharArr arr = m_featureBase->m_kernel->loadUCharArray(group, name, dims);

    // Check if load was successfull. Channels should always
    // have a dimensionality of [width x n]. So dim.size() 
    // has to be 2.
    if(arr != nullptr)
    {
        if(dims.size() != 2)
        {
            std::cout << timestamp << "ChannelIO.load(): Trying to load data with dim = " 
                      << dims.size() << "." << std::endl;
            std::cout << timestamp << "Should be 2." << std::endl;
            return false;
        }
        channel = Channel<unsigned char>(dims[1], dims[0], arr);
        return true;
    }

    return false;
}

template<typename FeatureBase>
bool ChannelIO<FeatureBase>::load(  
    std::string group,
    std::string name,
    Channel<double>& channel)
{
    std::vector<size_t> dims;
    doubleArr arr = m_featureBase->m_kernel->loadDoubleArray(group, name, dims);

    // Check if load was successfull. Channels should always
    // have a dimensionality of [width x n]. So dim.size() 
    // has to be 2.
    if(arr != nullptr)
    {
        if(dims.size() != 2)
        {
            std::cout << timestamp << "ChannelIO.load(): Trying to load data with dim = " 
                      << dims.size() << "." << std::endl;
            std::cout << timestamp << "Should be 2." << std::endl;
            return false;
        }
        channel = Channel<double>(dims[1], dims[0], arr);
        return true;
    }

    return false;
}

template<typename FeatureBase>
bool ChannelIO<FeatureBase>::load(  
    std::string group, 
    std::string name,
    Channel<int>& channel)
{
    std::vector<size_t> dims;
    intArr arr = m_featureBase->m_kernel->loadIntArray(group, name, dims);

    // Check if load was successfull. Channels should always
    // have a dimensionality of [width x n]. So dim.size() 
    // has to be 2.
    if(arr != nullptr)
    {
        if(dims.size() != 2)
        {
            std::cout << timestamp << "ChannelIO.load(): Trying to load data with dim = " 
                      << dims.size() << "." << std::endl;
            std::cout << timestamp << "Should be 2." << std::endl;
            return false;
        }
        channel = Channel<int>(dims[1], dims[0], arr);
        return true;
    }

    return false;
}

template<typename FeatureBase>
bool ChannelIO<FeatureBase>::load(  
    std::string group,
    std::string name,
    Channel<uint16_t>& channel)
{
    std::vector<size_t> dims;
    uint16Arr arr = m_featureBase->m_kernel->loadDoubleArray(group, name, dims);

    // Check if load was successfull. Channels should always
    // have a dimensionality of [width x n]. So dim.size() 
    // has to be 2.
    if(arr != nullptr)
    {
        if(dims.size() != 2)
        {
            std::cout << timestamp << "ChannelIO.load(): Trying to load data with dim = " 
                      << dims.size() << "." << std::endl;
            std::cout << timestamp << "Should be 2." << std::endl;
            return false;
        }
        channel = Channel<uint16_t>(dims[1], dims[0], arr);
        return true;
    }

    return false;
}


// SAVE
template<typename FeatureBase>
void ChannelIO<FeatureBase>::_save(  
    std::string group, 
    std::string name, 
    const Channel<float>& channel) const
{
    std::vector<size_t> dims(2);
    dims[0] = channel.width();
    dims[1] = channel.numElements();
    m_featureBase->m_kernel->saveFloatArray(group, name, dims, channel.dataPtr());
}

template<typename FeatureBase>
void ChannelIO<FeatureBase>::_save( std::string group,
                std::string name,
                const Channel<unsigned char>& channel) const
{
    std::vector<size_t> dims(2);
    dims[0] = channel.width();
    dims[1] = channel.numElements();
    m_featureBase->m_kernel->saveUCharArray(group, name, dims, channel.dataPtr());
}

template<typename FeatureBase>
void ChannelIO<FeatureBase>::_save( std::string group,
            std::string name,
            const Channel<double>& channel) const
{
    std::vector<size_t> dims(2);
    dims[0] = channel.width();
    dims[1] = channel.numElements();
    m_featureBase->m_kernel->saveDoubleArray(group, name, dims, channel.dataPtr());
}

template<typename FeatureBase>
void ChannelIO<FeatureBase>::_save( std::string group,
            std::string name,
            const Channel<int>& channel) const
{
    std::vector<size_t> dims(2);
    dims[0] = channel.width();
    dims[1] = channel.numElements();
    m_featureBase->m_kernel->saveIntArray(group, name, dims, channel.dataPtr());
}

template<typename FeatureBase>
void ChannelIO<FeatureBase>::_save( std::string group,
            std::string name,
            const Channel<uint16_t>& channel) const
{
    std::vector<size_t> dims(2);
    dims[0] = channel.width();
    dims[1] = channel.numElements();
    m_featureBase->m_kernel->saveUInt16Array(group, name, dims, channel.dataPtr());
}


} // namespace lvr2