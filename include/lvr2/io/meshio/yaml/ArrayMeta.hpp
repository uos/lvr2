#pragma once
#include <lvr2/io/meshio/ArrayMeta.hpp>

namespace YAML
{

template<>
struct convert<lvr2::meshio::ArrayMeta>
{
    static Node encode(const lvr2::meshio::ArrayMeta& array_meta)
    {
        
        YAML::Node node;
        node["entity"]      = array_meta.entity;
        node["type"]        = array_meta.type;
        node["data_type"]   = array_meta.data_type;
        node["shape"]       = array_meta.shape;

        return node;
    }

    static bool decode(const Node& node, lvr2::meshio::ArrayMeta& meta)
    {
        using lvr2::meshio::ArrayMeta;

        // Check if fields exist
        if (!node["entity"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<ArrayMeta> - decode] " 
                            << "Node has no tag 'entity'." << std::endl;
            return false;
        }

        if (!node["type"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<ArrayMeta> - decode] " 
                            << "Node has no tag 'type'." << std::endl;
            return false;    
        }

        if (!node["data_type"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<ArrayMeta> - decode] " 
                            << "Node has no tag 'data_type'." << std::endl;
            return false;
        }

        if (!node["shape"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<ArrayMeta> - decode] " 
                            << "Node has no tag 'shape'." << std::endl;
            return false;
        }

        // Check if they have the right value
        if (node["entity"].as<std::string>() != ArrayMeta::entity)
        {
            std::cout << lvr2::timestamp << "[YAML::convert<ArrayMeta> - decode] " 
                            << "entity '" << node["entity"] << "' is not '" 
                            << ArrayMeta::entity <<  "'." << std::endl;
            return false;
        }

        if (node["type"].as<std::string>() != ArrayMeta::type)
        {
            std::cout << lvr2::timestamp << "[YAML::convert<ArrayMeta> - decode] " 
                            << "type '" << node["type"] << "' is not '" 
                            << ArrayMeta::type <<  "'." << std::endl;
            return false;
        }

        meta.data_type = node["data_type"].as<std::string>();
        meta.shape = node["shape"].as<std::vector<size_t>>();

        return true;
    }
};

} // namespace YAML