
namespace lvr2 {

template<typename T>
boost::shared_array<T> DataIO::load()
{
    boost::shared_array<T> ret;
    
    movePosition(sizeof(Header) + m_header.JSON_BYTES);

    std::cout << "Load " << m_header.DATA_BYTES / sizeof(T) << " elements " << std::endl;
    ret.reset(new T[m_header.DATA_BYTES / sizeof(T)]);
    m_file.read(reinterpret_cast<char*>(&ret[0]), m_header.DATA_BYTES);
    m_pos += m_header.DATA_BYTES;
    return ret;
}

template<typename T>
boost::shared_array<T> DataIO::load(std::vector<size_t>& shape)
{
    boost::shared_array<T> ret;
    shape.clear();

    YAML::Node meta = loadMeta();
    shape = meta["SHAPE"].as<std::vector<size_t> >();

    movePosition(sizeof(Header) + m_header.JSON_BYTES);

    ret.reset(new T[m_header.DATA_BYTES / sizeof(T)]);
    m_file.read(reinterpret_cast<char*>(&ret[0]), m_header.DATA_BYTES);
    
    m_pos += m_header.DATA_BYTES;
    
    return ret;
}

template<typename T>
void DataIO::save(
    const std::vector<size_t>& shape, 
    const boost::shared_array<T>& data)
{
    Header header;
    header.MAGIC[0] = 'd';
    header.MAGIC[1] = 'a';
    header.MAGIC[2] = 't';
    header.MAGIC[3] = 'a';
    header.VERSION = 1;

    header.DATA_BYTES = sizeof(T);
    for(size_t i=0; i<shape.size(); i++)
    {
        header.DATA_BYTES *= shape[i];
    }

    // write Json with yaml-cpp
    YAML::Emitter emitter;
    emitter << YAML::DoubleQuoted << YAML::Flow;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "TYPE" << YAML::Value << dataIOTypeName<T>();
    emitter << YAML::Key << "SHAPE";
    emitter << YAML::Value << YAML::BeginSeq;
    for(size_t i=0; i<shape.size(); i++)
    {
        emitter << shape[i];
    }
    emitter << YAML::EndSeq;
    emitter << YAML::EndMap;
    header.JSON_BYTES = emitter.size();

    // std::cout << "Write Header" << std::endl;
    // std::cout << header << std::endl;
    // HEADER 24 bytes
    movePosition(0);
    m_file.clear();
    m_file.write(reinterpret_cast<const char*>(&header), sizeof(Header));
    m_file.write(emitter.c_str(), header.JSON_BYTES);
    m_file.write(reinterpret_cast<const char*>(&data[0]), header.DATA_BYTES);

    m_pos += sizeof(Header) + sizeof(header.JSON_BYTES) + sizeof(header.DATA_BYTES);
}

} // namespace lvr2