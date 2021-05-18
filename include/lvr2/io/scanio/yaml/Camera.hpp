
#ifndef LVR2_IO_YAML_CAMERA_HPP
#define LVR2_IO_YAML_CAMERA_HPP

#include <sstream>

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "Matrix.hpp"
#include "CameraModels.hpp"

namespace YAML {

/**
 * YAML-CPPs convert specialization
 * 
 * example: 
 */

// WRITE SCAN PARTIALLY
template<>
struct convert<lvr2::Camera> 
{

    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::Camera& camera) {
        
        Node node;
        node["type"] = lvr2::Camera::type;
        node["kind"] = lvr2::Camera::kind;
        node["name"] = camera.name;
        node["transformation"] = camera.transformation;
        node["model"] = camera.model;

        return node;
    }

    static bool decode(const Node& node, lvr2::Camera& camera) 
    {
        // Check if we are reading camera information
        if(!node["type"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<Camera> - decode] "
                     << "Camera meta has no key 'type'" << std::endl; 
            return false;
        }

        if(node["type"].as<std::string>() != lvr2::Camera::type)
        {
            std::cout << lvr2::timestamp << "[YAML::convert<Camera> - decode] " 
                        << "Nodes type '" << node["type"].as<std::string>()
                        << "' is not '" <<  lvr2::Camera::type << "'" << std::endl; 
            return false;
        }

        if(!node["kind"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<Camera> - decode] "
                     << "WARNING: Sensor has no key 'kind'. Assuming this sensor to by of kind "  << lvr2::Camera::kind << std::endl;
        } else {
            if(node["kind"].as<std::string>() != lvr2::Camera::kind)
            {
                std::cout << lvr2::timestamp << "[YAML::convert<Camera> - decode] " 
                            << "Nodes kind '" << node["kind"].as<std::string>()
                            << "' is not '" <<  lvr2::Camera::kind << "'" << std::endl; 
                return false;
            }
        }

        if(node["name"])
        {
            camera.name = node["name"].as<std::string>();
        }
        else
        {
            camera.name = "";
        }

        if(node["transformation"])
        {
            camera.transformation = node["transformation"].as<lvr2::Transformd>();
        }

        if(node["model"])
        {
            camera.model = node["model"].as<decltype(camera.model)>();
        }

        return true;
    }
};

}  // namespace YAML

#endif // LVR2_IO_YAML_PINHOLECAMERAMODEL_IO_HPP

