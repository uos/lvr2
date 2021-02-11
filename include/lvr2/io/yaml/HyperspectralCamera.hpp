
#ifndef LVR2_IO_YAML_HYPERSPECTRALCAMERAMETA_IO_HPP
#define LVR2_IO_YAML_HYPERSPECTRALCAMERAMETA_IO_HPP

#include <yaml-cpp/yaml.h>
#include "Matrix.hpp"
#include "CameraModels.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace YAML
{

/**
 * YAML-CPPs convert specialization
 *
 * example:
 */

// WRITE HYPERSPECTRALCAMERA PARTIALLY
template <>
struct convert<lvr2::HyperspectralCamera>
{

    /**
     * Encode Eigen matrix to yaml.
     */
    static Node encode(const lvr2::HyperspectralCamera& camera)
    {
        Node node;

        node["type"] = lvr2::HyperspectralCamera::type;
        node["kind"] = lvr2::HyperspectralCamera::kind;
        node["name"] = camera.name;
        node["transformation"] = camera.transformation;
        node["model"] = camera.model;

        return node;
    }

    static bool decode(const Node& node, lvr2::HyperspectralCamera& camera)
    {
        if(!node["type"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralCamera> - decode] "
                     << "HyperspectralCamera meta has no key 'type'" << std::endl; 
            return false;
        }

        if (node["type"].as<std::string>() != lvr2::HyperspectralCamera::type)
        {
            // different hierarchy level
            std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralCamera> - decode] " 
                        << "Nodes type '" << node["type"].as<std::string>()
                        << "' is not '" <<  lvr2::HyperspectralCamera::type << "'" << std::endl; 
            return false;
        }

        if(!node["kind"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralCamera> - decode] "
                     << "WARNING: Sensor has no key 'kind'. Assuming this sensor to by of kind "  << lvr2::Camera::kind << std::endl;
        } else {
            if(node["kind"].as<std::string>() != lvr2::HyperspectralCamera::kind)
            {
                std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralCamera> - decode] " 
                            << "Nodes kind '" << node["kind"].as<std::string>()
                            << "' is not '" <<  lvr2::HyperspectralCamera::kind << "'" << std::endl; 
                return false;
            }
        }

        if(node["name"])
        {
            camera.name = node["name"].as<decltype(camera.name)>();
        }

        if(node["model"])
        {
            camera.model= node["model"].as<decltype(camera.model)>();
        }

        return true;
    }
};

template <>
struct convert<lvr2::HyperspectralPanorama>
{

    /**
     * Encode Eigen matrix to yaml.
     */
    static Node encode(const lvr2::HyperspectralPanorama& pano)
    {
        Node node;

        node["type"] = lvr2::HyperspectralCamera::type;
        node["kind"] = lvr2::HyperspectralCamera::kind;
        node["transformation"] = pano.transformation;
        // node["model"] = pano.model;

        return node;
    }

    static bool decode(const Node& node, lvr2::HyperspectralPanorama& pano)
    {
        // if(!node["type"])
        // {
        //     std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralCamera> - decode] "
        //              << "HyperspectralCamera meta has no key 'type'" << std::endl; 
        //     return false;
        // }

        // if (node["type"].as<std::string>() != lvr2::HyperspectralCamera::type)
        // {
        //     // different hierarchy level
        //     std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralCamera> - decode] " 
        //                 << "Nodes type '" << node["type"].as<std::string>()
        //                 << "' is not '" <<  lvr2::HyperspectralCamera::type << "'" << std::endl; 
        //     return false;
        // }

        // if(!node["kind"])
        // {
        //     std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralCamera> - decode] "
        //              << "WARNING: Sensor has no key 'kind'. Assuming this sensor to by of kind "  << lvr2::Camera::kind << std::endl;
        // } else {
        //     if(node["kind"].as<std::string>() != lvr2::HyperspectralCamera::kind)
        //     {
        //         std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralCamera> - decode] " 
        //                     << "Nodes kind '" << node["kind"].as<std::string>()
        //                     << "' is not '" <<  lvr2::HyperspectralCamera::kind << "'" << std::endl; 
        //         return false;
        //     }
        // }

        // if(node["name"])
        // {
        //     camera.name = node["name"].as<decltype(camera.name)>();
        // }

        // if(node["model"])
        // {
        //     camera.model= node["model"].as<decltype(camera.model)>();
        // }

        return true;
    }
};

} // namespace YAML

#endif // LVR2_IO_YAML_HYPERSPECTRALCAMERAMETA_IO_HPP
