#include "lvr2/util/Tuple.hpp"

namespace lvr2 
{

namespace baseio
{

template<typename BaseIO>
template<typename T> 
ChannelOptional<T> ChannelIO<BaseIO>::load(
    std::string group, std::string name) const
{   
    ChannelOptional<T> ret;

    if constexpr(FileKernel::ImplementedTypes::contains<T>())
    {
        ret = loadFundamental<T>(group, name);
    }
    else
    {
        // NOT IMPLEMENTED TYPE TO READ

        T tmp_obj;
        size_t tmp_size;
        ucharArr tmp_buffer = byteEncode(tmp_obj, tmp_size);

        if (tmp_buffer)
        {
            if (byteDecode<T>(&tmp_buffer[0], tmp_size))
            {
                ret = loadCustom<T>(group, name);
            }
            else
            {
                lvr2::logout::get() << lvr2::warning << "[ChannelIO] Type not implemented for " << group << "/" << name << lvr2::endl;
            }
        }
        else
        {
            lvr2::logout::get() << lvr2::warning << "[ChannelIO] Type not implemented for " << group << "/" << name << lvr2::endl;
        }
    }

    return ret;
}

template<typename BaseIO>
template<typename T>
void ChannelIO<BaseIO>::save(
    std::string group,
    std::string name,
    const Channel<T>& channel) const
{
    if constexpr(FileKernel::ImplementedTypes::contains<T>())
    {
        saveFundamental(group, name, channel);
    }
    else
    {

        if (channel.width() == 1 && channel.numElements() > 0)
        {
            // could be dynamic channel
            // check if serialization is implemented for the specific type
            saveCustom(group, name, channel);
        }
        else
        {
            // NOT IMPLEMENTED TYPE TO WRITE
            lvr2::logout::get() << lvr2::warning << "[ChannelIO] Type not implemented for " << group << "/" << name << lvr2::endl;
        }
    }
}

template<typename BaseIO>
template<typename T>
void ChannelIO<BaseIO>::save(
    const size_t& scanPosNo,
    const size_t& lidarNo,
    const size_t& scanNo,
    const std::string& channelName,
    const Channel<T>& channel
) const
{
    auto Dgen = m_baseIO->m_description;
    Description d = Dgen->position(scanPosNo);
    d = Dgen->lidar(d, lidarNo);
    d = Dgen->scan(d, scanNo);
}

template<typename BaseIO>
template<typename T>
void ChannelIO<BaseIO>::saveCustom(
    std::string group,
    std::string name,
    const Channel<T>& channel) const
{
    // lvr2::logout::get() << "[ChannelIO - saveCustom ]" << lvr2::endl;
    size_t tmp_size;
    T tmp_obj;
    boost::shared_array<unsigned char> buffer = byteEncode(tmp_obj, tmp_size);

    if (buffer)
    {
        // saving is possible via uchar buffer
        // Determine size of shared array
        size_t total_size = sizeof(size_t);
        for (size_t i = 0; i < channel.numElements(); i++)
        {
            size_t buffer_size;
            buffer = byteEncode(channel[i][0], buffer_size);
            // +1 because of size_t per element
            total_size += buffer_size + sizeof(size_t);
        }

        boost::shared_array<unsigned char> data(new unsigned char[total_size]);
        unsigned char *data_ptr = &data[0];

        size_t Npoints = channel.numElements();
        unsigned char *Npointsc = reinterpret_cast<unsigned char *>(&Npoints);
        memcpy(data_ptr, Npointsc, sizeof(size_t));
        data_ptr += sizeof(size_t);

        // lvr2::logout::get() << "Write Meta: " << Npoints << " points" << lvr2::endl;

        // Filling shared_array with data
        for (size_t i = 0; i < channel.numElements(); i++)
        {
            size_t buffer_size;
            buffer = byteEncode(channel[i][0], buffer_size);
            unsigned char *bsize = reinterpret_cast<unsigned char *>(&buffer_size);

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
        m_baseIO->m_kernel->saveUCharArray(group, name, dims, data);
    }
    else
    {
        lvr2::logout::get() << lvr2::warning << "[ChannelIO] Type not implemented for " << group << "/" << name << lvr2::endl;
    }
}

template<typename BaseIO>
std::vector<size_t> ChannelIO<BaseIO>::loadDimensions(
    std::string groupName, std::string datasetName) const
{
    std::vector<size_t> dims;
    YAML::Node node;
    m_baseIO->m_kernel->loadMetaYAML(groupName, datasetName, node);
    for(auto it = node["dims"].begin(); it != node["dims"].end(); ++it)
    {
        dims.push_back(it->as<size_t>());
    }

    return dims;
}

// PROTECTED

// LOADER
template <typename BaseIO>
template <typename T>
ChannelOptional<T> ChannelIO<BaseIO>::loadFundamental(
    std::string group,
    std::string name) const
{
    ChannelOptional<T> ret;

    if constexpr (FileKernel::ImplementedTypes::contains<T>())
    {
        std::vector<size_t> dims;
        boost::shared_array<T> arr = m_baseIO->m_kernel->template loadArray<T>(group, name, dims);

        if (arr)
        {
            if (dims.size() != 2)
            {
                lvr2::logout::get() << lvr2::warning << "[ChannelIO] Trying to load data with dim = "
                          << dims.size() << ". Should be 2." << lvr2::endl;
                return ret;
            }
            Channel<T> c(dims[0], dims[1], arr);
            ret = c;
            return ret;
        }
    }
    else
    {
        // WARNINGS
    }
    return ret;
}

template<typename BaseIO>
template<typename T>
ChannelOptional<T> ChannelIO<BaseIO>::loadCustom(
    std::string group,
    std::string name
) const
{
    ChannelOptional<T> ret;
    // deserialize
    
    // found readable custom type
    std::vector<size_t> dims;
    ucharArr buffer = m_baseIO->m_kernel->loadUCharArray(group, name, dims);

    unsigned char *data_ptr = &buffer[0];
    size_t Npoints = *reinterpret_cast<size_t *>(data_ptr);
    data_ptr += sizeof(size_t);

    Channel<T> cd(Npoints, 1);

    for (size_t i = 0; i < Npoints; i++)
    {
        const size_t elem_size = *reinterpret_cast<const size_t *>(data_ptr);
        data_ptr += sizeof(size_t);
        auto dataopt = byteDecode<T>(data_ptr, elem_size);
        if (dataopt)
        {
            cd[i][0] = *dataopt;
        }
        else
        {
            // could not load object of type T
            return ret;
        }
        data_ptr += elem_size;
    }

    ret = cd;

    return ret;
}

// SAVE
template<typename BaseIO>
template<typename T>
void ChannelIO<BaseIO>::saveFundamental(
    std::string group,
    std::string name,
    const Channel<T>& channel) const
{
    // lvr2::logout::get() << "ChannelIO - saveFundamental" << lvr2::endl;
    if constexpr (FileKernel::ImplementedTypes::contains<T>())
    {
        std::vector<size_t> dims(2);
        dims[0] = channel.numElements();
        dims[1] = channel.width();
        // lvr2::logout::get() << "Save Channel " << dims[0] << "x" << dims[1] << lvr2::endl;
        m_baseIO->m_kernel->template saveArray<T>(group, name, dims, channel.dataPtr());
    }
    else
    {
        // TODO: Error or Warning?
        std::stringstream ss;
        ss << "Kernel does not support channels of type '" << channel.typeName() << "'";
        lvr2::logout::get() << ss.str() << lvr2::endl;
        throw std::runtime_error(ss.str());
    }
}

} // namespace scanio

} // namespace lvr2