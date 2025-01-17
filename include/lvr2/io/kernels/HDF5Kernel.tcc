namespace lvr2
{

template<typename T>
ChannelOptional<T> HDF5Kernel::loadChannelOptional(
    const std::string& groupName,
    const std::string& datasetName) const  
{
    ChannelOptional<T> ret;

    if(hdf5util::exist(m_hdf5File, groupName))
    {
        HighFive::Group g = hdf5util::getGroup(m_hdf5File, groupName, false);
        ret = loadChannelOptional<T>(g, datasetName);
    } 

    return ret;
}

template<typename T>
ChannelOptional<T> HDF5Kernel::loadChannelOptional(
    HighFive::Group& g,
    const std::string& datasetName) const
{
    ChannelOptional<T> ret;

    if (m_hdf5File && m_hdf5File->isValid())
    {
        if (g.exist(datasetName))
        {
            HighFive::DataSet dataset = g.getDataSet(datasetName);
            std::vector<size_t> dim = dataset.getSpace().getDimensions();

            size_t elementCount = 1;
            for (auto e : dim)
                elementCount *= e;

            if (elementCount)
            {
                ret = Channel<T>(dim[0], dim[1]);
                dataset.read(ret->dataPtr().get());
            }
        }
    }
    else
    {
        throw std::runtime_error("[Hdf5 - ChannelIO]: Hdf5 file not open.");
    }

    return ret;
}

template <typename T>
boost::shared_array<T> HDF5Kernel::loadArray(
    const std::string &groupName,
    const std::string &datasetName, 
    size_t &size) const
{
    boost::shared_array<T> ret;

    HighFive::Group g = hdf5util::getGroup(
        m_hdf5File,
        groupName,
        false
    );

    std::vector<size_t> dim;
    ret = load<T>(g, datasetName, dim);

    size = 1;
    for (auto cur : dim)
    {
        size *= cur;
    }

    return ret;
}

template<typename T>
boost::shared_array<T> HDF5Kernel::loadArray(
    const std::string& groupName, 
    const std::string& datasetName, 
    std::vector<size_t>& dim) const
{
    boost::shared_array<T> ret;
    HighFive::Group g = hdf5util::getGroup(m_hdf5File, groupName);

    if(m_hdf5File && m_hdf5File->isValid())
    {
        if (g.exist(datasetName))
        {
            HighFive::DataSet dataset = g.getDataSet(datasetName);
            dim = dataset.getSpace().getDimensions();

            size_t elementCount = 1;
            for (auto e : dim)
                elementCount *= e;

            if(elementCount)
            {
                ret = boost::shared_array<T>(new T[elementCount]);

                dataset.read(ret.get());
            }
        }
    } 
    else 
    {
        throw std::runtime_error("[Hdf5 - ArrayIO]: Hdf5 file not open.");
    }

    return ret;
}

template<typename T>
void HDF5Kernel::saveArray(
    const std::string& groupName,
    const std::string& datasetName,
    const size_t& size,
    const boost::shared_array<T> data) const
{
    std::vector<size_t> dim = {size, 1};
    save(groupName, datasetName, dim,  data);
}


template<typename T>
static std::vector<hsize_t> autoComputeChunkSize(
    const std::vector<size_t> dims,
    size_t chunkBytes)
{    
    std::vector<hsize_t> chunk_sizes(dims.size(), 1);

    size_t elems_per_chunk = chunkBytes / sizeof(T);
    size_t current_elems_per_chunk = 1;
    
    size_t dim_id = dims.size() - 1;

    while(dim_id >= 0 && dims[dim_id] * current_elems_per_chunk <= elems_per_chunk) 
    {
        current_elems_per_chunk *= dims[dim_id];
        chunk_sizes[dim_id] = dims[dim_id];
        // If all dimensions were processed return early, otherwise dim_id will wrap because its unsigned
        // this can lead to invalid memory access
        if (dim_id == 0)
        {
            return chunk_sizes;
        }

        dim_id -= 1;
    }

    chunk_sizes[dim_id] = elems_per_chunk / current_elems_per_chunk;
    return chunk_sizes;
}

template<typename T> 
void HDF5Kernel::saveArray(
    const std::string& groupName, 
    const std::string& datasetName,
    const vector<size_t>& dim,
    const boost::shared_array<T> data) const
{
    HighFive::Group g = hdf5util::getGroup(m_hdf5File, groupName, true);
    
    // std::cout << "[HDF5Kernel - saveArray]" << std::endl;
    if(m_hdf5File && m_hdf5File->isValid())
    {
        HighFive::DataSpace dataSpace(dim);
        HighFive::DataSetCreateProps properties;

        if(m_config.compressionLevel > 0)
        {
            std::vector<hsize_t> chunkSizes = autoComputeChunkSize<T>(dim, 10 * 1024);
            properties.add(HighFive::Chunking(chunkSizes));
            properties.add(HighFive::Deflate(m_config.compressionLevel));
        }
        
        std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<T>(
            g, datasetName, dataSpace, properties
        );

        const T* ptr = data.get();
        dataset->write_raw(ptr);
        m_hdf5File->flush();
    } 
    else 
    {
        throw std::runtime_error("[HDF5Kernel - saveArray]: Hdf5 file not open.");
    }
}


template <typename T>
bool HDF5Kernel::getChannel(const std::string group, const std::string name, boost::optional<AttributeChannel<T>>& channel)  const
{
    // TODO check group for vertex / face attribute and set flag in hdf5 channel
    HighFive::Group g = hdf5util::getGroup(m_hdf5File, "channels");
    if(m_hdf5File && m_hdf5File->isValid())
    {
        if(g.exist(name))
        {
            HighFive::DataSet dataset = g.getDataSet(name);
            std::vector<size_t> dim = dataset.getSpace().getDimensions();

            size_t elementCount = 1;
            for (auto e : dim)
            {
                elementCount *= e;
            }
               
            if(elementCount)
            {
                channel = Channel<T>(dim[0], dim[1]);
                dataset.read(channel->dataPtr().get());
            }
        }
    }
    else
    {
        throw std::runtime_error("[Hdf5 - ChannelIO]: Hdf5 file not open.");
    }
    return true;
}


template <typename T>
bool HDF5Kernel::addChannel(
    const std::string group, const std::string name, 
    const AttributeChannel<T>& channel)  const
{
    if(m_hdf5File && m_hdf5File->isValid())
    {
        HighFive::DataSpace dataSpace({channel.numElements(), channel.width()});
        HighFive::DataSetCreateProps properties;

        // TODO check group for vertex / face attribute and set flag in hdf5 channel
        HighFive::Group g = hdf5util::getGroup(m_hdf5File, "channels");

        std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<T>(
                g, name, dataSpace, properties);

        const T* ptr = channel.dataPtr().get();
        dataset->write_raw(ptr);
        m_hdf5File->flush();
        std::cout << timestamp << " Added attribute \"" << name << "\" to group \"" << group
                  << "\" to the given HDF5 file!" << std::endl;
    }
    else
    {
        throw std::runtime_error("[Hdf5IO - ChannelIO]: Hdf5 file not open.");
    }
    return true;
}


// R == 0
template<typename VariantT, int R, typename std::enable_if<R == 0, void>::type* = nullptr>
void saveVChannel(
    const VariantT& vchannel,
    const HDF5Kernel* channel_io,
    HighFive::Group& group,
    std::string name)
{
    if(R == vchannel.type())
    {
        channel_io->save(group, name, vchannel.template extract<typename VariantT::template type_of_index<R> >() );
    } 
    else 
    {
        std::cout << "[VariantChannelIO] WARNING: Nothing was saved" << std::endl;
    }
}

// R != 0
template<typename VariantT, int R, typename std::enable_if<R != 0, void>::type* = nullptr>
void saveVChannel(
    const VariantT& vchannel,
    const HDF5Kernel* channel_io,
    HighFive::Group& group,
    std::string name)
{
    if(R == vchannel.type())
    {
        channel_io->save(group, name, vchannel.template extract<typename VariantT::template type_of_index<R> >() );
    } 
    else 
    {
        saveVChannel<VariantT, R-1>(vchannel, channel_io, group, name);
    }
}

template <typename T>
void HDF5Kernel::save(std::string groupName,
          std::string datasetName,
          const Channel<T> &channel) const
{
    HighFive::Group g = hdf5util::getGroup(m_hdf5File, groupName);
    save(g, datasetName, channel);
}

template <typename T>
void HDF5Kernel::save(HighFive::Group &g,
          std::string datasetName,
          const Channel<T> &channel) const
{
    if constexpr(hdf5util::H5AllowedTypes::contains<T>())
    {
        if(m_hdf5File && m_hdf5File->isValid())
        {
            std::vector<size_t > dims = {channel.numElements(), channel.width()};

            HighFive::DataSpace dataSpace(dims);
            HighFive::DataSetCreateProps properties;

            if(m_config.compressionLevel > 0)
            {
                std::vector<hsize_t> chunkSizes;
                chunkSizes.assign(dims.begin(), dims.end()); 
                properties.add(HighFive::Chunking(chunkSizes));
                properties.add(HighFive::Deflate(m_config.compressionLevel));
            }
    
            std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<T>(
                g, datasetName, dataSpace, properties
            );

            const T* ptr = channel.dataPtr().get();
            dataset->write_raw(ptr);
            m_hdf5File->flush();

            std::string sensor_type = "Channel";
            hdf5util::setAttribute<std::string>(*dataset, "sensor_type", sensor_type);
        } 
        else 
        {
            throw std::runtime_error("[Hdf5IO - ChannelIO]: Hdf5 file not open.");
        }
    } else {
        std::cout << "[Hdf5IO - save]: Type not supported by Hdf5" << std::endl;
    }
}

template<typename ...Tp>
void HDF5Kernel::save(
    std::string groupName,
    std::string datasetName,
    const VariantChannel<Tp...>& vchannel) const
{
    // TODO
    HighFive::Group g = hdf5util::getGroup(
        m_hdf5File,
        groupName,
        true
    );

    this->save<Tp...>(g, datasetName, vchannel);
}


template<typename ...Tp>
void HDF5Kernel::save(
    HighFive::Group& group,
    std::string datasetName,
    const VariantChannel<Tp...>& vchannel) const
{
    saveVChannel<VariantChannel<Tp...>, VariantChannel<Tp...>::num_types-1>(vchannel, this, group, datasetName);
}

// R == 0
template<typename VariantChannelT, int R, typename std::enable_if<R == 0, void>::type* = nullptr>
boost::optional<VariantChannelT> loadVChannel(
    HighFive::DataType dtype,
    const HDF5Kernel* channel_io,
    HighFive::Group& group,
    std::string name)
{
    boost::optional<VariantChannelT> ret;
    if(dtype == HighFive::AtomicType<typename VariantChannelT::template type_of_index<R> >())
    {
        auto channel = channel_io->template loadChannelOptional<typename VariantChannelT::template type_of_index<R> >(group, name);
        if(channel) {
            ret = *channel;
        }
        return ret;
    } else {
        return ret;
    }
}

// R != 0
template<typename VariantChannelT, int R, typename std::enable_if<R != 0, void>::type* = nullptr>
boost::optional<VariantChannelT> loadVChannel(
    HighFive::DataType dtype,
    const HDF5Kernel* channel_io,
    HighFive::Group& group,
    std::string name)  
{
    boost::optional<VariantChannelT> ret;
    if(dtype == HighFive::AtomicType<typename VariantChannelT::template type_of_index<R> >())
    {
        boost::optional<VariantChannelT> ret;
        auto loaded_channel = channel_io->loadChannelOptional<typename VariantChannelT::template type_of_index<R> >(group, name);
        if(loaded_channel)
        {
            ret = *loaded_channel;
        }
        return ret;
    } 
    else 
    {
        return loadVChannel<VariantChannelT, R-1>(dtype, channel_io, group, name);
    }
}


template<typename VariantChannelT>
boost::optional<VariantChannelT> HDF5Kernel::loadDynamic(
    HighFive::DataType dtype,
    HighFive::Group& group,
    std::string name) const
{
    return loadVChannel<VariantChannelT, VariantChannelT::num_types-1>(
        dtype, this, group, name);
}


template<typename VariantChannelT>
boost::optional<VariantChannelT> HDF5Kernel::load(
    std::string groupName,
    std::string datasetName) const 
{
    boost::optional<VariantChannelT> ret;

    if(hdf5util::exist(m_hdf5File, groupName))
    {
        HighFive::Group g = hdf5util::getGroup(m_hdf5File, groupName, false);
        ret = this->load<VariantChannelT>(g, datasetName);
    } else {
        std::cout << "[VariantChannelIO] WARNING: Group " << groupName << " not found." << std::endl;
    }

    return ret;
}


template<typename VariantChannelT>
boost::optional<VariantChannelT> HDF5Kernel::load(
    HighFive::Group& group,
    std::string datasetName) const
{
    boost::optional<VariantChannelT> ret;

    std::unique_ptr<HighFive::DataSet> dataset;

    try {
        dataset = std::make_unique<HighFive::DataSet>(
            group.getDataSet(datasetName)
        );
    } catch(HighFive::DataSetException& ex) {
        std::cout << "[VariantChannelIO] WARNING: Dataset " << datasetName << " not found." << std::endl;
    }

    if(dataset)
    {
        // name is dataset
        ret = loadDynamic<VariantChannelT>(dataset->getDataType(), group, datasetName);
    }

    return ret;
}


template<typename VariantChannelT>
boost::optional<VariantChannelT> HDF5Kernel::loadVariantChannel(
    std::string groupName,
    std::string datasetName) const
{
    return load<VariantChannelT>(groupName, datasetName);
}

template<typename T>
cv::Mat HDF5Kernel::createMat(const std::vector<size_t>& dims) const
{
    cv::Mat ret;

    // single channel type
    int cv_type = cv::DataType<T>::type;

    if(dims.size() > 2)
    {
        cv_type += (dims[2]-1) * 8;
    }

    if(dims.size() > 1)
    {
        ret = cv::Mat(dims[0], dims[1], cv_type);
    } 
    else 
    {
        ret = cv::Mat(dims[0], 1, cv_type);
    }

    return ret;
}

} // namespace lvr2