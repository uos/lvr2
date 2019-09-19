namespace lvr2 {

namespace hdf5features {

template<typename Derived>
template<typename T>
ChannelOptional<T> ChannelIO<Derived>::load(std::string groupName,
    std::string datasetName)
{
    ChannelOptional<T> ret;

    if(hdf5util::exist(m_file_access->m_hdf5_file, groupName))
    {
        HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, groupName, false);
        ret = load<T>(g, datasetName);
    } 

    return ret;
}

template<typename Derived>
template<typename T>
ChannelOptional<T> ChannelIO<Derived>::load(
    HighFive::Group& g,
    std::string datasetName)
{
    ChannelOptional<T> ret;

    if(m_file_access->m_hdf5_file && m_file_access->m_hdf5_file->isValid())
    {
        if(g.exist(datasetName))
        {
            HighFive::DataSet dataset = g.getDataSet(datasetName);
            std::vector<size_t> dim = dataset.getSpace().getDimensions();
            
            size_t elementCount = 1;
            for (auto e : dim)
                elementCount *= e;

            if(elementCount)
            {
                ret = Channel<T>(dim[0], dim[1]);
                dataset.read(ret->dataPtr().get());
            }
        }
    } else {
        throw std::runtime_error("[Hdf5 - ArrayIO]: Hdf5 file not open.");
    }

    return ret;
}

template<typename Derived>
template<typename T>
ChannelOptional<T> ChannelIO<Derived>::loadChannel(std::string groupName,
    std::string datasetName)
{
    return load<T>(groupName, datasetName);
}

template<typename Derived>
template<typename T>
void ChannelIO<Derived>::save(std::string groupName,
        std::string datasetName,
        const Channel<T>& channel)
{
    HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, groupName);
    save(g, datasetName, channel);
}

template<typename Derived>
template<typename T>
void ChannelIO<Derived>::save(HighFive::Group& g,
    std::string datasetName,
    const Channel<T>& channel)
{
    std::vector<hsize_t> chunks = {channel.numElements(), channel.width()};
    save(g, datasetName, channel, chunks);
}

template<typename Derived>
template<typename T>
void ChannelIO<Derived>::save(HighFive::Group& g,
    std::string datasetName,
    const Channel<T>& channel,
    std::vector<hsize_t>& chunkSizes)
{
    if(m_file_access->m_hdf5_file && m_file_access->m_hdf5_file->isValid())
    {
        std::vector<size_t > dims = {channel.numElements(), channel.width()};

        HighFive::DataSpace dataSpace(dims);
        HighFive::DataSetCreateProps properties;

        if(m_file_access->m_chunkSize)
        {
            for(size_t i = 0; i < chunkSizes.size(); i++)
            {
                if(chunkSizes[i] > dims[i])
                {
                    chunkSizes[i] = dims[i];
                }
            }
            properties.add(HighFive::Chunking(chunkSizes));
        }
        if(m_file_access->m_compress)
        {
            //properties.add(HighFive::Shuffle());
            properties.add(HighFive::Deflate(9));
        }
   
        std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<T>(
            g, datasetName, dataSpace, properties
        );

        const T* ptr = channel.dataPtr().get();
        dataset->write(ptr);
        m_file_access->m_hdf5_file->flush();
    } else {
        throw std::runtime_error("[Hdf5IO - ChannelIO]: Hdf5 file not open.");
    }
}

} // namespace hdf5features

} // namespace hdf5features