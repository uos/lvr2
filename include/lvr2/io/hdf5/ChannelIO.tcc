#include "Hdf5Util.hpp"
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
        throw std::runtime_error("[Hdf5 - ChannelIO]: Hdf5 file not open.");
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



template<typename Derived>
template <typename T>
bool ChannelIO<Derived>::getChannel(const std::string group, const std::string name, boost::optional<AttributeChannel<T>>& channel)
{
    // TODO check group for vertex / face attribute and set flag in hdf5 channel
    HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, "channels");
    if(m_file_access->m_hdf5_file && m_file_access->m_hdf5_file->isValid())
    {
        if(g.exist(name))
        {
            HighFive::DataSet dataset = g.getDataSet(name);
            std::vector<size_t> dim = dataset.getSpace().getDimensions();

            size_t elementCount = 1;
            for (auto e : dim)
                elementCount *= e;

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

template<typename Derived>
template <typename T>
bool ChannelIO<Derived>::addChannel(const std::string group, const std::string name, const AttributeChannel<T>& channel)
{
    if(m_file_access->m_hdf5_file && m_file_access->m_hdf5_file->isValid())
    {
        HighFive::DataSpace dataSpace({channel.numElements(), channel.width()});
        HighFive::DataSetCreateProps properties;

        if(m_file_access->m_chunkSize)
        {
            properties.add(HighFive::Chunking({channel.numElements(), channel.width()}));
        }
        if(m_file_access->m_compress)
        {
            //properties.add(HighFive::Shuffle());
            properties.add(HighFive::Deflate(9));
        }

        // TODO check group for vertex / face attribute and set flag in hdf5 channel
        HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, "channels");

        std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<T>(
                g, name, dataSpace, properties);

        const T* ptr = channel.dataPtr().get();
        dataset->write(ptr);
        m_file_access->m_hdf5_file->flush();
        std::cout << timestamp << " Added attribute \"" << name << "\" to group \"" << group
                  << "\" to the given HDF5 file!" << std::endl;
    }
    else
    {
        throw std::runtime_error("[Hdf5IO - ChannelIO]: Hdf5 file not open.");
    }
    return true;
}

template<typename Derived>
bool ChannelIO<Derived>::getChannel(const std::string group, const std::string name, FloatChannelOptional& channel)
{
    return getChannel<float>(group, name, channel);
}

template<typename Derived>
bool ChannelIO<Derived>::getChannel(const std::string group, const std::string name, IndexChannelOptional& channel)
{
    return getChannel<unsigned int>(group, name, channel);
}

template<typename Derived>
bool ChannelIO<Derived>::getChannel(const std::string group, const std::string name, UCharChannelOptional& channel)
{
    return getChannel<unsigned char>(group, name, channel);
}

template<typename Derived>
bool ChannelIO<Derived>::addChannel(const std::string group, const std::string name, const FloatChannel& channel)
{
    return addChannel<float>(group, name, channel);
}

template<typename Derived>
bool ChannelIO<Derived>::addChannel(const std::string group, const std::string name, const IndexChannel& channel)
{
    return addChannel<unsigned int>(group, name, channel);
}

template<typename Derived>
bool ChannelIO<Derived>::addChannel(const std::string group, const std::string name, const UCharChannel& channel)
{
    return addChannel<unsigned char>(group, name, channel);
}

} // namespace hdf5features

} // namespace lvr2