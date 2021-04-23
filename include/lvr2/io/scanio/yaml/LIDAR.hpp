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
        node["type"] = lvr2::LIDAR::type;
        node["kind"] = lvr2::LIDAR::kind;
        node["transformation"] = lidar.transformation;
        return node;
    }

    static bool decode(const Node& node, lvr2::LIDAR& lidar) 
    {
        // if(!node["type"])
        // {
        //     std::cout << "[YAML::convert<LIDAR> - decode] 'type' Tag not found." << std::endl;
        //     return false;
        // }

        if(node["type"] && node["type"].as<std::string>() != lvr2::LIDAR::type) 
        {
            std::cout << "[YAML::convert<LIDAR> - decode] Try to load " << node["type"].as<std::string>() << " as " << lvr2::LIDAR::type << std::endl;
            return false;
        }


        if(!node["kind"]) // optional kind
        {
            std::cout << "[YAML::convert<LIDAR> - decode] WARNING: 'kind' Tag not found. Assuming this meta to be of kind " << lvr2::LIDAR::kind << std::endl;
            // return false;
        } else {
            if(node["kind"].as<std::string>() != lvr2::LIDAR::kind) 
            {
                std::cout << "[YAML::convert<LIDAR> - decode] Try to load " << node["kind"].as<std::string>() << " as " << lvr2::LIDAR::kind << std::endl;
                return false;
            }
        }

        if(node["transformation"])
        {
            lidar.transformation = node["transformation"].as<lvr2::Transformd>();
        } else {
            lidar.transformation = lvr2::Transformd::Identity();
        }

        return true;
    }
};

} // namespace YAML

#endif // LVR2_IO_YAML_LIDAR_HPP
