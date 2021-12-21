#ifndef LVR2_IO_YAML_LIDAR_HPP
#define LVR2_IO_YAML_LIDAR_HPP

#include "Matrix.hpp"
#include "AABB.hpp"

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

        if(lidar.boundingBox)
        {
            node["aabb"] = *lidar.boundingBox;
        }

        return node;
    }

    static bool decode(const Node& node, lvr2::LIDAR& lidar) 
    {
        if (!YAML_UTIL::ValidateEntityAndType(node, 
            "lidar", 
            lvr2::LIDAR::entity, 
            lvr2::LIDAR::type))
        {
            return false;
        }

        // std::cout << "name" << std::endl;
        if(node["transformation"])
        {
            lidar.transformation = node["transformation"].as<lvr2::Transformd>();
        } else {
            lidar.transformation = lvr2::Transformd::Identity();
        }

        // std::cout << "name" << std::endl;
        if(node["name"])
        {
            lidar.name = node["name"].as<std::string>();
        }

        // std::cout << "model" << std::endl;
        if(node["model"])
        {
            lidar.model = node["model"].as<lvr2::SphericalModel>();
        } else {
            // defaults
            lidar.model.range[0] = 0.0;
            
        }

        // std::cout << "aabb" << std::endl;
        if(node["aabb"])
        {
            lidar.boundingBox = node["aabb"].as<lvr2::BoundingBox<lvr2::BaseVector<float> > >();
        }

        // std::cout << "return" << std::endl;

        return true;
    }
};

} // namespace YAML

#endif // LVR2_IO_YAML_LIDAR_HPP
