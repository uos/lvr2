#ifndef LVR2_IO_YAML_LIDAR_HPP
#define LVR2_IO_YAML_LIDAR_HPP

#include "lvr2/io/YAML.hpp"

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

        if(node["transformation"])
        {
            try {
                lidar.transformation = node["transformation"].as<lvr2::Transformd>();
            } catch(const YAML::TypedBadConversion<lvr2::Transformd>& ex) {
                std::cerr << "[YAML - LIDAR - decode] ERROR: Could not decode 'transformation': "
                    << node["transformation"] << " as Transformd" << std::endl; 
                return false;
            }
        } else {
            lidar.transformation = lvr2::Transformd::Identity();
        }

        if(node["name"])
        {
            try {
                lidar.name = node["name"].as<std::string>();
            } catch(const YAML::TypedBadConversion<std::string>& ex) {
                std::cerr << "[YAML - LIDAR - decode] ERROR: Could not decode 'name': "
                    << node["name"] << " as string" << std::endl; 
                return false;
            }
        }

        if(node["model"])
        {
            try {
                lidar.model = node["model"].as<lvr2::SphericalModel>();
            } catch(const YAML::TypedBadConversion<lvr2::SphericalModel>& ex) {
                std::cerr << "[YAML - LIDAR - decode] ERROR: Could not decode 'model': "
                    << node["model"] << " as SphericalModel" << std::endl; 
                return false;
            }
        } else {
            lidar.model.range[0] = 0.0;
        }

        if(node["aabb"])
        {
            try {
                lidar.boundingBox = node["aabb"].as<lvr2::BoundingBox<lvr2::BaseVector<float> > >();
            } catch(const YAML::TypedBadConversion<lvr2::BoundingBox<lvr2::BaseVector<float> > >& ex) {
                std::cerr << "[YAML - LIDAR - decode] ERROR: Could not decode 'aabb': "
                    << node["aabb"] << " as BoundingBox" << std::endl; 
                return false;
            }
        }

        return true;
    }
};

} // namespace YAML

#endif // LVR2_IO_YAML_LIDAR_HPP
