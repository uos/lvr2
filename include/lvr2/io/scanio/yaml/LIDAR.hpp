#ifndef LVR2_IO_YAML_LIDAR_HPP
#define LVR2_IO_YAML_LIDAR_HPP

#include "Matrix.hpp"

namespace YAML {  

/**
 * YAML-CPPs convert specialization
 * 
 * example: 
 */

// WRITE LIDAR META
template <>
struct convert<lvr2::LIDAR> 
{
    static Node encode(const lvr2::LIDAR& lidar) {
        Node node;
        node["entity"] = lvr2::LIDAR::entity;
        node["type"] = lvr2::LIDAR::type;
        node["transformation"] = lidar.transformation;
        node["name"] = lidar.name;
        node["model"] = lidar.model;
        return node;
    }

    static bool decode(const Node& node, lvr2::LIDAR& lidar) 
    {
        // Check for entity field
        if(!node["entity"])
        {
            std::cout << "[YAML::convert<LIDAR> - decode] 'entity' Tag not found." << std::endl;
            return false;
        }
        if(node["entity"].as<std::string>() != lvr2::LIDAR::entity) 
        {
            std::cout << "[YAML::convert<LIDAR> - decode] Try to load " << node["entity"].as<std::string>() << " as " << lvr2::LIDAR::entity << std::endl;
            return false;
        }

        // Check for typ field
        if(!node["type"])
        {
            std::cout << "[YAML::convert<LIDAR> - decode] 'type' Tag not found." << std::endl;
            return false;
        }
        if(node["type"].as<std::string>() != lvr2::LIDAR::type) 
        {
            std::cout << "[YAML::convert<LIDAR> - decode] Try to load " << node["type"].as<std::string>() << " as " << lvr2::LIDAR::type << std::endl;
            return false;
        }

        if(node["transformation"])
        {
            lidar.transformation = node["transformation"].as<lvr2::Transformd>();
        } else {
            lidar.transformation = lvr2::Transformd::Identity();
        }

        if(node["name"])
        {
            lidar.name = node["name"].as<std::string>();
        }

        if(node["model"])
        {
            lidar.model = node["model"].as<lvr2::SphericalModel>();
        }

        return true;
    }
};

} // namespace YAML

#endif // LVR2_IO_YAML_LIDAR_HPP
