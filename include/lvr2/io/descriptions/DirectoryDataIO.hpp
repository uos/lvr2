#ifndef LVR2_DIRECTORY_DATA_IO_HPP
#define LVR2_DIRECTORY_DATA_IO_HPP

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

struct DataIOHeader
{
    char MAGIC[4];
    int VERSION;
    long unsigned int JSON_BYTES;
    long unsigned int DATA_BYTES;
};

std::ostream& operator<<(std::ostream& os, const DataIOHeader& header);



template<typename T>
void dataIOsave(
    std::string filename, 
    const std::vector<size_t>& shape, 
    const boost::shared_array<T>& data)
{
    DataIOHeader header;
    // const char MAGIC[4] = {'d', 'a', 't', 'a'};
    header.MAGIC[0] = 'd';
    header.MAGIC[1] = 'a';
    header.MAGIC[2] = 't';
    header.MAGIC[3] = 'a';
    header.VERSION = 1;

    // std::cout << "Saving Header of size " << sizeof(DataIOHeader) << std::endl;

    header.DATA_BYTES = sizeof(T);
    for(size_t i=0; i<shape.size(); i++)
    {
        header.DATA_BYTES *= shape[i];
    }

    YAML::Emitter emitter;
    emitter << YAML::DoubleQuoted << YAML::Flow;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "type" << YAML::Value << dataIOTypeName<T>();
    emitter << YAML::Key << "shape";
    emitter << YAML::Value << YAML::BeginSeq;
    for(size_t i=0; i<shape.size(); i++)
    {
        emitter << shape[i];
    }
    emitter << YAML::EndSeq;
    emitter << YAML::EndMap;
    header.JSON_BYTES = emitter.size();


    // now write
    std::ofstream fout(filename);

    // HEADER 24 bytes
    fout.write(reinterpret_cast<const char*>(&header), sizeof(DataIOHeader));
    fout.write(emitter.c_str(), header.JSON_BYTES);
    fout.write(reinterpret_cast<const char*>(&data[0]), header.DATA_BYTES);
    fout.close();
}

template<typename T>
boost::shared_array<T> dataIOload(
    std::string filename, 
    std::vector<size_t>& shape)
{
    boost::shared_array<T> ret; 

    std::ifstream fin(filename);
    
    /// LOAD HEADER
    DataIOHeader header;
    fin.read(reinterpret_cast<char*>(&header), sizeof(DataIOHeader));

    char * json_meta = new char[header.JSON_BYTES];
    fin.read(json_meta, header.JSON_BYTES);

    YAML::Node meta;
    meta = YAML::Load(json_meta);

    shape = meta["shape"].as<std::vector<size_t> >();
    
    size_t nelements = 1;
    for(auto elem : shape)
    {
        nelements *= elem;
    }
    
    ret.reset(new T[nelements]);
    fin.read(reinterpret_cast<char*>(&ret[0]), header.DATA_BYTES);
    fin.close();

    delete[] json_meta;
    return ret;
}

DataIOHeader dataIOloadHeader(std::string filename);

} // namespace lvr2

#endif // LVR2_DIRECTORY_DATA_IO_HPP
