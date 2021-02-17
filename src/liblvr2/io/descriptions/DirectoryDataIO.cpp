#include "lvr2/io/descriptions/DirectoryDataIO.hpp"

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

std::ostream& operator<<(std::ostream& os, const DataIOHeader& header)
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

DataIOHeader dataIOloadHeader(std::string filename)
{
    std::ifstream fin;
    fin.open(filename, std::ios::binary | std::ios::in);
    /// LOAD HEADER
    DataIOHeader header;
    fin.read(reinterpret_cast<char*>(&header), sizeof(DataIOHeader));
    fin.close();

    return header;
}

} // namespace lvr2