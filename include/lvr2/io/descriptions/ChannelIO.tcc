namespace lvr2 
{

template<typename FeatureBase>
template<typename T> 
ChannelOptional<T> ChannelIO<FeatureBase>::load(
    std::string group, std::string name) const
{   
    ChannelOptional<T> ret;
    Channel<T> c;

    if constexpr(std::is_same<T, float>::value ) {
        if(load(group, name, c)){ret = c;}
    } else if constexpr(std::is_same<T, unsigned char>::value ) {
        if(load(group, name, c)){ret = c;}
    } else if constexpr(std::is_same<T, double>::value ) {
        if(load(group, name, c)){ret = c;}
    } else if constexpr(std::is_same<T, int>::value ) {
        if(load(group, name, c)){ret = c;}
    } else if constexpr(std::is_same<T, uint16_t>::value ) {
        if(load(group, name, c)){ret = c;}
    } else {
        // NOT IMPLEMENTED TYPE TO READ

        T tmp_obj;
        size_t tmp_size;
        ucharArr tmp_buffer = serialize(tmp_obj, tmp_size);
        
        if(tmp_buffer)
        {
            // deserialize
            if(deserialize<T>(&tmp_buffer[0], tmp_size))
            {
                // found readable custom type

                std::cout << "Reading possible!" << std::endl;
                std::vector<size_t> dims;
                ucharArr buffer = m_featureBase->m_kernel->loadUCharArray(group, name, dims);
                
                unsigned char* data_ptr = &buffer[0];
                size_t Npoints = *reinterpret_cast<size_t*>(data_ptr);
                data_ptr += sizeof(size_t);

                std::cout << "Loading Dynamic Channel for " << Npoints << " points " << std::endl;

                Channel<T> cd(Npoints, 1);

                for(size_t i=0; i<Npoints; i++)
                {
                    std::cout << "Point " << i << std::endl;
                    const size_t elem_size = *reinterpret_cast<const size_t*>(data_ptr);
                    std::cout << "  elem_size: " << elem_size << std::endl; 
                    data_ptr += sizeof(size_t);
                    cd[i][0] = *deserialize<T>(data_ptr, elem_size);
                    data_ptr += elem_size;
                }

                ret = c;
                


            } else {
                std::cout << "[ChannelIO] Type not implemented for " << group << "/" << name << std::endl;
            }
        } else {
            std::cout << "[ChannelIO] Type not implemented for " << group << "/" << name << std::endl;
        }

        
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

        if(channel.width() == 1 && channel.numElements() > 0)
        {
            // could be dynamic channel
            // check if serialization is implemented for the specific type
            size_t tmp_size;
            T tmp_obj;
            boost::shared_array<unsigned char> buffer = serialize(tmp_obj, tmp_size);
            


            if(buffer)
            {
                // saving is possible via uchar buffer


                // Determine size of shared array
                size_t total_size = sizeof(size_t);
                for(size_t i = 0; i<channel.numElements(); i++)
                {
                    size_t buffer_size;
                    buffer = serialize(channel[i][0], buffer_size);
                    // +1 because of size_t per element
                    total_size += buffer_size + sizeof(size_t);
                }

                boost::shared_array<unsigned char> data(new unsigned char[total_size]);
                unsigned char* data_ptr = &data[0];
                
                size_t Npoints = channel.numElements();
                unsigned char* Npointsc = reinterpret_cast<unsigned char*>(&Npoints);
                memcpy(data_ptr, Npointsc, sizeof(size_t));
                data_ptr += sizeof(size_t);

                std::cout << "Write Meta: " << Npoints << " points" << std::endl;

                // Filling shared_array with data
                for(size_t i = 0; i<channel.numElements(); i++)
                {
                    size_t buffer_size;
                    buffer = serialize(channel[i][0], buffer_size);
                    unsigned char* bsize = reinterpret_cast<unsigned char*>(&buffer_size);
            
                    // write buffer size
                    memcpy(data_ptr, bsize, sizeof(size_t));
                    data_ptr += sizeof(size_t);
                    // write buffer
                    memcpy(data_ptr, &buffer[0], buffer_size);
                    data_ptr += buffer_size;
                }

                std::vector<size_t> dims(2);
                dims[0] = total_size;
                dims[1] = 1;
                m_featureBase->m_kernel->saveUCharArray(group, name, dims, data);
            } else {
                std::cout << "[ChannelIO] Type not implemented for " << group << "/" << name << std::endl;
            }
        } else {
            // NOT IMPLEMENTED TYPE TO WRITE
            std::cout << "[ChannelIO] Type not implemented for " << group << "/" << name << std::endl;
        }
        
    }
}

template<typename FeatureBase>
std::vector<size_t> ChannelIO<FeatureBase>::loadDimensions(
    std::string groupName, std::string datasetName) const
{
    std::vector<size_t> dims;
    YAML::Node node;
    m_featureBase->m_kernel->loadMetaYAML(groupName, datasetName, node);
    for(auto it = node["dims"].begin(); it != node["dims"].end(); ++it)
    {
        dims.push_back(it->as<size_t>());
    }

    return dims;
}

// PROTECTED



template<typename FeatureBase>
bool ChannelIO<FeatureBase>::load(  
    std::string group, std::string name,
    Channel<float>& channel) const
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
            std::cout << timestamp << "[ChannelIO] Trying to load data with dim = " 
                      << dims.size() << ". Should be 2." << std::endl;
            return false;
        }
        channel = Channel<float>(dims[0], dims[1], arr);
        return true;
    } 

    return false;
}

template<typename FeatureBase>
bool ChannelIO<FeatureBase>::load(  
    std::string group, std::string name,
    Channel<unsigned char>& channel) const
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
            std::cout << timestamp << "[ChannelIO] Trying to load data with dim = " 
                      << dims.size() << ". Should be 2." << std::endl;
            return false;
        }
        channel = Channel<unsigned char>(dims[0], dims[1], arr);
        return true;
    }

    return false;
}

template<typename FeatureBase>
bool ChannelIO<FeatureBase>::load(  
    std::string group, std::string name,
    Channel<double>& channel) const
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
            std::cout << timestamp << "[ChannelIO] Trying to load data with dim = " 
                      << dims.size() << ". Should be 2." << std::endl;
            return false;
        }
        channel = Channel<double>(dims[0], dims[1], arr);
        return true;
    }

    return false;
}

template<typename FeatureBase>
bool ChannelIO<FeatureBase>::load(  
    std::string group, std::string name,
    Channel<int>& channel) const
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
            std::cout << timestamp << "[ChannelIO] Trying to load data with dim = " 
                      << dims.size() << ". Should be 2." << std::endl;
            return false;
        }
        channel = Channel<int>(dims[0], dims[1], arr);
        return true;
    }

    return false;
}

template<typename FeatureBase>
bool ChannelIO<FeatureBase>::load(  
    std::string group, std::string name,
    Channel<uint16_t>& channel) const
{
    std::vector<size_t> dims;
    uint16Arr arr = m_featureBase->m_kernel->loadUInt16Array(group, name, dims);

    // Check if load was successfull. Channels should always
    // have a dimensionality of [width x n]. So dim.size() 
    // has to be 2.
    if(arr != nullptr)
    {
        if(dims.size() != 2)
        {
            std::cout << timestamp << "[ChannelIO] Trying to load data with dim = " 
                      << dims.size() << ". Should be 2." << std::endl;
            return false;
        }
        channel = Channel<uint16_t>(dims[0], dims[1], arr);
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
    dims[0] = channel.numElements();
    dims[1] = channel.width();
    m_featureBase->m_kernel->saveFloatArray(group, name, dims, channel.dataPtr());
}

template<typename FeatureBase>
void ChannelIO<FeatureBase>::_save( std::string group,
                std::string name,
                const Channel<unsigned char>& channel) const
{
    std::vector<size_t> dims(2);
    dims[0] = channel.numElements();
    dims[1] = channel.width();
    m_featureBase->m_kernel->saveUCharArray(group, name, dims, channel.dataPtr());
}

template<typename FeatureBase>
void ChannelIO<FeatureBase>::_save( std::string group,
            std::string name,
            const Channel<double>& channel) const
{
    std::vector<size_t> dims(2);
    dims[0] = channel.numElements();
    dims[1] = channel.width();
    m_featureBase->m_kernel->saveDoubleArray(group, name, dims, channel.dataPtr());
}

template<typename FeatureBase>
void ChannelIO<FeatureBase>::_save( std::string group,
            std::string name,
            const Channel<int>& channel) const
{
    std::vector<size_t> dims(2);
    dims[0] = channel.numElements();
    dims[1] = channel.width();
    m_featureBase->m_kernel->saveIntArray(group, name, dims, channel.dataPtr());
}

template<typename FeatureBase>
void ChannelIO<FeatureBase>::_save( std::string group,
            std::string name,
            const Channel<uint16_t>& channel) const
{
    std::vector<size_t> dims(2);
    dims[0] = channel.numElements();
    dims[1] = channel.width();
    m_featureBase->m_kernel->saveUInt16Array(group, name, dims, channel.dataPtr());
}


} // namespace lvr2