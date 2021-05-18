#include "lvr2/io/scanio/DirectoryDataIO.hpp"

namespace lvr2 {

template<>
std::string dataIOTypeName<unsigned char>()
{
    return "UNSGINED_BYTE";
}

template<>
std::string dataIOTypeName<unsigned short>()
{
    return "UNSIGNED_SHORT";
}

template<>
std::string dataIOTypeName<unsigned int>()
{
    return "UNSIGNED_INT";
}

template<>
std::string dataIOTypeName<unsigned long int>()
{
    return "UNSIGNED_LONG";
}

template<>
std::string dataIOTypeName<char>()
{
    return "BYTE";
}

template<>
std::string dataIOTypeName<short>()
{
    return "SHORT";
}

template<>
std::string dataIOTypeName<int>()
{
    return "INT";
}

template<>
std::string dataIOTypeName<long int>()
{
    return "LONG";
}

template<>
std::string dataIOTypeName<float>()
{
    return "FLOAT";
}

template<>
std::string dataIOTypeName<double>()
{
    return "DOUBLE";
}

std::ostream& operator<<(std::ostream& os, const DataIO::Header& header)
{
    os << "DataIOHeader" << std::endl;
    // a string needs a termination character '\0'
    char tmp[5] = {header.MAGIC[0], header.MAGIC[1], header.MAGIC[2], header.MAGIC[3], '\0'};
    os << "- MAGIC: " << tmp << std::endl;
    os << "- VERSION: " << header.VERSION << std::endl;
    os << "- JSON_BYTES: " << header.JSON_BYTES << std::endl;
    os << "- DATA_BYTES: " << header.DATA_BYTES << std::endl;
    return os;
}

DataIO::DataIO(std::string filename, std::ios_base::openmode ios_mode)
:m_pos(0)
{
    m_file.open(filename, ios_mode | std::ios::binary);
    m_header = loadHeader();
}

DataIO::~DataIO()
{
    m_file.close();
}

DataIO::Header DataIO::loadHeader()
{
    Header ret;
    movePosition(0);

    m_file.read(reinterpret_cast<char*>(&ret), sizeof(Header));
    m_pos += sizeof(Header);
    return ret;
}

YAML::Node DataIO::loadMeta()
{
    YAML::Node ret;

    movePosition(sizeof(Header));

    char * json_meta = new char[m_header.JSON_BYTES + 1];
    json_meta[m_header.JSON_BYTES] = '\0';
    m_file.read(json_meta, m_header.JSON_BYTES);

    ret = YAML::Load(json_meta);
    m_pos += m_header.JSON_BYTES;
    
    delete[] json_meta;
    return ret;
}

std::vector<size_t> DataIO::loadShape()
{
    return loadMeta()["SHAPE"].as<std::vector<size_t> >();
}

std::string DataIO::loadType()
{
    return loadMeta()["TYPE"].as<std::string>();
}

} // namespace lvr2