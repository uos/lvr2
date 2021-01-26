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
        node["kind"] = boost::typeindex::type_id<lvr2::LIDAR>().pretty_name();
        node["transformation"] = lidar.transformation;
        return node;
    }

    static bool decode(const Node& node, lvr2::LIDAR& lidar) 
    {
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

        return true;
    }
};

} // namespace YAML

#endif // LVR2_IO_YAML_LIDAR_HPP
