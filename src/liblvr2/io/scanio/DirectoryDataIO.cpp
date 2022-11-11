#include "lvr2/io/scanio/DirectoryDataIO.hpp"

#include <unordered_map>

namespace lvr2 {

template<>
std::string dataIOTypeName<bool>()
{
    return "|u1";
}

template<>
std::string dataIOTypeName<uint8_t>()
{
    return "|u1";
}

template<>
std::string dataIOTypeName<uint16_t>()
{
    return "<u2";
}

template<>
std::string dataIOTypeName<uint32_t>()
{
    return "<u4";
}

template<>
std::string dataIOTypeName<uint64_t>()
{
    return "<u8";
}

/**
 * @brief Additional Conversion for char required. char != signed char
 * There are three distinct basic character types: char, signed char and unsigned char.
 * - https://stackoverflow.com/questions/16503373/difference-between-char-and-signed-char-in-c
 * 
 * @tparam  
 * @return std::string 
 */
template<>
std::string dataIOTypeName<char>()
{
    return "|i1";
}

template<>
std::string dataIOTypeName<int8_t>()
{
    return "|i1";
}

template<>
std::string dataIOTypeName<int16_t>()
{
    return "<i2";
}

template<>
std::string dataIOTypeName<int32_t>()
{
    return "<i4";
}

template<>
std::string dataIOTypeName<int64_t>()
{
    return "<i8";
}

template<>
std::string dataIOTypeName<float>()
{
    return "<f4";
}

template<>
std::string dataIOTypeName<double>()
{
    return "<f8";
}



/**
 * @brief Add Backwards compatibility here
 * 
 * 0: V1 -> V2
 * 1: V2 -> V3
 * 
 */
static std::vector<std::unordered_map<std::string, std::string> > compat_maps = {
    { // V1 -> V2
        {"UNSGINED_BYTE", "|u1"},
        {"UNSIGNED_SHORT", "<u2"},
        {"UNSIGNED_INT", "<u4"},
        {"UNSIGNED_LONG", "<u8"},
        {"BYTE", "|i1"},
        {"SHORT", "<i2"},
        {"INT", "<i4"},
        {"LONG", "<i8"},
        {"FLOAT", "<f4"},
        {"DOUBLE", "<f8"},
        {"UNDEFINED", "UNDEFINED"}
    }
};



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

DataIO::DataIO(std::string filename, 
    std::ios_base::openmode ios_mode)
:m_pos(0)
{
    m_file.open(filename, ios_mode | std::ios::binary);
    if(!m_file)
    {
        if((ios_mode & std::ios::in) 
            && (ios_mode & std::ios::out))
        {
            m_file.open(filename, std::ios::out | std::ios::binary);

            if(!m_file)
            {
                std::cout << "DataIO: could not create file" << std::endl;
            } 

            m_file.close();

            m_file.open(filename, ios_mode | std::ios::binary);
        }

        if(!m_file)
        {
            std::cout << "DataIO: Could not open '" << filename << "'" << std::endl; 
            throw std::runtime_error("DataIO: Could not open file");
        }
    }

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
    Header header = loadHeader();

    if(header.VERSION > version())
    {
        std::cout << "DataIO - maximum readable version: V" << version() << ". Found V" << header.VERSION << std::endl;
        throw std::runtime_error("Cannot read data file. Version to high");
    }

    YAML::Node ret;
    movePosition(sizeof(Header));

    char * json_meta = new char[m_header.JSON_BYTES + 1];
    json_meta[m_header.JSON_BYTES] = '\0';
    m_file.read(json_meta, m_header.JSON_BYTES);

    ret = YAML::Load(json_meta);

    // loading old version
    if(header.VERSION < version())
    {
        for(size_t i = header.VERSION - 1; i < version() - 1; i++)
        {
            std::string old_type = ret["TYPE"].as<std::string>();
            if(compat_maps[i].find(old_type) == compat_maps[i].end())
            {
                throw std::runtime_error("Could not find old type in compatibility map");
            }
            ret["TYPE"] = compat_maps[i][old_type];
        }
    }

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

int DataIO::version() const
{
    return 2;
}

} // namespace lvr2