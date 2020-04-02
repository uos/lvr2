namespace lvr2 {

template<typename FeatureBase>
FloatChannelOptionsal ChannelIO<FeatureBase>::loadFloatChannel(std::string groupName, std::string datasetName)
{
    FloatChannelOptional ret;
    
    std::vector<size_t> dims;
    floatArr arr = m_featureBase->m_kernel->loadArray<T>(groupName, datasetName, dims);

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
            return boost::none;
        }
        ret = Channel<T>(dims[1], dim[0], arr);
        return ret;
    }

    return boost::none;
}


template<typename FeatureBase>
template<typename T>
void ChannelIO<FeatureBase>::saveChannel(std::string groupName,
        std::string datasetName,
        const Channel<T>& channel)
{
    std::vector<size_t> dims(2);
    dims[0] = channel.width();
    dims[1] = channel.numElements();

    m_featureBase->kernel(groupName, datasetName, channel.dataPtr(), dims);
}



template<typename FeatureBase>
template <typename T>
bool ChannelIO<FeatureBase>::getChannel(const std::string group, const std::string name, boost::optional<AttributeChannel<T>>& channel)
{
    channel = load(group, name);
    if(channel)
    {
        return true;
    }
    else
    {
        return false;
    }
}

template<typename FeatureBase>
template <typename T>
bool ChannelIO<FeatureBase>::addChannel(const std::string group, const std::string name, const AttributeChannel<T>& channel)
{
    save(group, name, channel);
    return true;
}

template<typename FeatureBase>
bool ChannelIO<FeatureBase>::getChannel(const std::string group, const std::string name, FloatChannelOptional& channel)
{
    return getChannel<float>(group, name, channel);
}

template<typename FeatureBase>
bool ChannelIO<FeatureBase>::getChannel(const std::string group, const std::string name, IndexChannelOptional& channel)
{
    return getChannel<unsigned int>(group, name, channel);
}

template<typename FeatureBase>
bool ChannelIO<FeatureBase>::getChannel(const std::string group, const std::string name, UCharChannelOptional& channel)
{
    return getChannel<unsigned char>(group, name, channel);
}

template<typename FeatureBase>
bool ChannelIO<FeatureBase>::addChannel(const std::string group, const std::string name, const FloatChannel& channel)
{
    return addChannel<float>(group, name, channel);
}

template<typename FeatureBase>
bool ChannelIO<FeatureBase>::addChannel(const std::string group, const std::string name, const IndexChannel& channel)
{
    return addChannel<unsigned int>(group, name, channel);
}

template<typename FeatureBase>
bool ChannelIO<FeatureBase>::addChannel(const std::string group, const std::string name, const UCharChannel& channel)
{
    return addChannel<unsigned char>(group, name, channel);
}

} // namespace lvr2