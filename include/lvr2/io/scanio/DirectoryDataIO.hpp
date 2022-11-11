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
std::string dataIOTypeName<bool>();

template<>
std::string dataIOTypeName<uint8_t>();

template<>
std::string dataIOTypeName<uint16_t>();

template<>
std::string dataIOTypeName<uint32_t>();

template<>
std::string dataIOTypeName<uint64_t>();

template<>
std::string dataIOTypeName<char>();

template<>
std::string dataIOTypeName<int8_t>();

template<>
std::string dataIOTypeName<int16_t>();

template<>
std::string dataIOTypeName<int32_t>();

template<>
std::string dataIOTypeName<int64_t>();

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
    
    DataIO(std::string filename, 
        std::ios_base::openmode ios_mode = std::ios::in | std::ios::out );

    ~DataIO();

    
    /**
     * @brief Load the header from file
     * 
     * @return Header 
     */
    Header loadHeader();

    /**
     * @brief Get all meta data as YAML node
     * 
     * @return YAML::Node 
     */
    YAML::Node loadMeta();

    /**
     * @brief Load the shape from file
     * 
     * @return std::vector<size_t> 
     */
    std::vector<size_t> loadShape();
    
    /**
     * @brief Returns type of the elements
     * 
     * if(dataIOTypeName<float>() == io.loadType())
     * {
     *    data = io.load<float>();
     * }
     * 
     * @return std::string 
     */
    std::string loadType();

    /**
     * @brief 
     * 
     * @tparam T 
     * @return boost::shared_array<T> 
     */
    template<typename T>
    boost::shared_array<T> load();

    /**
     * @brief Load data and write shape to "shape"
     * 
     * @tparam T  type of the data. Can be obtained by reading the meta data first
     * @param shape shape of the data
     * @return boost::shared_array<T>  returned data
     */
    template<typename T>
    boost::shared_array<T> load(std::vector<size_t>& shape);

    /**
     * @brief Save multidiomensional data of type T and shape "shape"
     * 
     * @tparam T typename of data elements
     * @param shape multidimensional shape
     * @param data data buffer
     */
    template<typename T>
    void save(const std::vector<size_t>& shape, 
              const boost::shared_array<T>& data);

    /**
     * @brief Version of the IO. Update in cpp to signal a version change
     * 
     * @return int the version
     */
    int version() const;

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
