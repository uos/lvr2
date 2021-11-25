
#ifndef LVR2_IO_YAML_CAMERA_HPP
#define LVR2_IO_YAML_CAMERA_HPP

#include <sstream>

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "Matrix.hpp"
#include "CameraModels.hpp"
#include "lvr2/io/scanio/yaml/Util.hpp"

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
        node["entity"] = lvr2::Camera::entity;
        node["type"] = lvr2::Camera::type;
        node["name"] = camera.name;
        node["transformation"] = camera.transformation;
        node["model"] = camera.model;

        return node;
    }

    static bool decode(const Node& node, lvr2::Camera& camera) 
    {
        // Check if 'entity' and 'type' Tags are valid
        if (!YAML_UTIL::ValidateEntityAndType(node, "Camera", lvr2::Camera::entity, lvr2::Camera::type))
        {
            return false;
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

