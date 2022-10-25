#ifndef DIRECTORYDATAIO
#define DIRECTORYDATAIO

#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <boost/shared_array.hpp>
#include <cstring>

namespace lvr2 {


template<typename T>
std::string dataIOTypeName();

template<typename T>
std::string dataIOTypeName()
{
    return "UNDEFINED";
}

template<>
std::string dataIOTypeName<unsigned char>();

template<>
std::string dataIOTypeName<unsigned short>();

template<>
std::string dataIOTypeName<unsigned int>();

template<>
std::string dataIOTypeName<unsigned long int>();

template<>
std::string dataIOTypeName<char>();

template<>
std::string dataIOTypeName<short>();

template<>
std::string dataIOTypeName<int>();

template<>
std::string dataIOTypeName<long int>();

template<>
std::string dataIOTypeName<float>();

template<>
std::string dataIOTypeName<double>();

class DataIO {
public:
    
    struct Header {
        char MAGIC[4];
        int VERSION;
        long unsigned int JSON_BYTES;
        long unsigned int DATA_BYTES;
    };
    
    DataIO(std::string filename, std::ios_base::openmode ios_mode = std::ios::in | std::ios::out );

    Header loadHeader();

    YAML::Node loadMeta();

    std::vector<size_t> loadShape();
    
    std::string loadType();

    template<typename T>
    boost::shared_array<T> load();

    template<typename T>
    boost::shared_array<T> load(std::vector<size_t>& shape);

    template<typename T>
    void save(const std::vector<size_t>& shape, 
              const boost::shared_array<T>& data);

    ~DataIO();

private:

    inline void movePosition(size_t byte)
    {
        if(m_pos < byte)
        {
            // offset if direction is positive
            m_file.seekg(byte - m_pos, std::ios_base::cur);
            m_pos = byte;
        } else if(m_pos > byte) {
            // set position relative to file begin
            m_file.seekg(byte, std::ios_base::beg);
            m_pos = byte;
        }
    }

    std::fstream m_file;
    Header m_header;
    size_t m_pos;
};

using DataIOPtr = std::shared_ptr<DataIO>;

std::ostream& operator<<(std::ostream& os, const DataIO::Header& header);

} // namespace lvr2

#include "DirectoryDataIO.tcc"

#endif // DIRECTORYDATAIO
