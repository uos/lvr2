
#ifndef LVR2_IO_YAML_HYPERSPECTRALCAMERAMETA_IO_HPP
#define LVR2_IO_YAML_HYPERSPECTRALCAMERAMETA_IO_HPP

#include <yaml-cpp/yaml.h>

#include "Matrix.hpp"
#include "CameraModels.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/scanio/yaml/Util.hpp"

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

        node["entity"] = lvr2::HyperspectralCamera::entity;
        node["type"] = lvr2::HyperspectralCamera::type;
        node["name"] = camera.name;
        node["transformation"] = camera.transformation;
        node["model"] = camera.model;

        return node;
    }

    static bool decode(const Node& node, lvr2::HyperspectralCamera& camera)
    {
        /*** Check for deprecated Tags and print warnings ***/
        if(node["kind"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralCamera> - decode] " 
                        << "WARNING: 'kind' Tag is no longer supported! " 
                        << "Please update your dataset to use 'entity' and 'type' Tags." << std::endl;
        } 
        if(node["sensor_type"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralCamera> - decode] "
                        << "Warning: 'sensor_type' Tag is no longer supported! "
                        << "Please update your dataset to use 'entity' and 'type' Tags." << std::endl;
        }
        /*** Continue parsing in case these Tags were redundant ***/

        // Check if 'entity' and 'type' Tags are valid
        if (!YAML_UTIL::ValidateEntityAndType(node, 
            "hyperspectral_camera", 
            lvr2::HyperspectralCamera::entity, 
            lvr2::HyperspectralCamera::type))
        {
            return false;
        }

        
        if(node["name"])
        {
            camera.name = node["name"].as<decltype(camera.name)>();
        }

        if(node["sensor_name"])
        {
            camera.name = node["sensor_name"].as<decltype(camera.name)>();
        }

        if(node["model"])
        {
            camera.model= node["model"].as<decltype(camera.model)>();
        } else {
            std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralCamera> - decode] "
                << "WARNING: Hyperspectral camera has no sensor model in meta file." << std::endl;
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

        node["entity"] = lvr2::HyperspectralPanorama::entity;
        node["type"] = lvr2::HyperspectralPanorama::type;
        node["transformation"] = pano.transformation;
        node["resolution"] = Load("[]");
        node["resolution"].push_back(pano.resolution[0]);
        node["resolution"].push_back(pano.resolution[1]);

        node["num_bands"] = pano.num_bands;
        node["frame_order"] = pano.frame_order;

        if(pano.model)
        {
            node["model"] = *pano.model;
        }

        // node["wavelength"] = Load("[]");
        // node["wavelength"].push_back(pano.wavelength[0]);
        // node["wavelength"].push_back(pano.wavelength[1]);

        return node;
    }

    static bool decode(const Node& node, lvr2::HyperspectralPanorama& pano)
    {
        /*** Check for deprecated Tags and print warnings ***/
        if(node["kind"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralPanorama> - decode] " 
                        << "WARNING: 'kind' Tag is no longer supported! " 
                        << "Please update your dataset to use 'entity' and 'type' Tags." << std::endl;
        } 
        if(node["sensor_type"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralPanorama> - decode] "
                        << "Warning: 'sensor_type' Tag is no longer supported! "
                        << "Please update your dataset to use 'entity' and 'type' Tags." << std::endl;
        }
        /*** Continue parsing in case these Tags were redundant ***/

        // Check if 'entity' and 'type' Tags are valid
        if (!YAML_UTIL::ValidateEntityAndType(node, 
            "HyperspectralPanorama", 
            lvr2::HyperspectralPanorama::entity, 
            lvr2::HyperspectralPanorama::type))
        {
            return false;
        }

        if(node["transformation"])
        {
            pano.transformation = node["transformation"].as<lvr2::Transformd>();
        }

        if(node["resolution"])
        {
            pano.resolution[0] = node["resolution"][0].as<unsigned int>();
            pano.resolution[1] = node["resolution"][1].as<unsigned int>();
        }

        if(node["num_bands"])
        {
            pano.num_bands = node["num_bands"].as<unsigned int>();
        }

        if(node["frame_order"])
        {
            pano.frame_order = node["frame_order"].as<std::string>();
        }

        if(node["model"])
        {
            pano.model = node["model"].as<lvr2::CylindricalModel>();
        }

        return true;
    }
};

template <>
struct convert<lvr2::HyperspectralPanoramaChannel>
{

    /**
     * Encode Eigen matrix to yaml.
     */
    static Node encode(const lvr2::HyperspectralPanoramaChannel& hchannel)
    {
        Node node;

        node["entity"] = lvr2::HyperspectralPanoramaChannel::entity;
        node["type"] = lvr2::HyperspectralPanoramaChannel::type;

        return node;
    }

    static bool decode(const Node& node, lvr2::HyperspectralPanoramaChannel& hchannel)
    {
        // Check if 'entity' and 'type' Tags are valid
        if (!YAML_UTIL::ValidateEntityAndType(node, 
            "HyperspectralPanoramaChannel", 
            lvr2::HyperspectralPanoramaChannel::entity, 
            lvr2::HyperspectralPanoramaChannel::type))
        {
            return false;
        }
        // if(!node["type"])
        // {
        //     std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralPanorama> - decode] "
        //              << "HyperspectralPanorama meta has no key 'type'" << std::endl; 
        //     return false;
        // }

        // if (node["type"].as<std::string>() != lvr2::HyperspectralPanorama::type)
        // {
        //     // different hierarchy level
        //     std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralPanorama> - decode] " 
        //                 << "Nodes type '" << node["type"].as<std::string>()
        //                 << "' is not '" <<  lvr2::HyperspectralPanorama::type << "'" << std::endl; 
        //     return false;
        // }

        // if(!node["kind"])
        // {
        //     std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralPanorama> - decode] "
        //              << "WARNING: Sensor has no key 'kind'. Assuming this sensor to by of kind "  << lvr2::Camera::kind << std::endl;
        // } else {
        //     if(node["kind"].as<std::string>() != lvr2::HyperspectralPanorama::kind)
        //     {
        //         std::cout << lvr2::timestamp << "[YAML::convert<HyperspectralPanorama> - decode] " 
        //                     << "Nodes kind '" << node["kind"].as<std::string>()
        //                     << "' is not '" <<  lvr2::HyperspectralPanorama::kind << "'" << std::endl; 
        //         return false;
        //     }
        // }

        // if(node["transformation"])
        // {
        //     pano.transformation = node["transformation"].as<lvr2::Transformd>();
        // }

        // if(node["resolution"])
        // {
        //     pano.resolution[0] = node["resolution"][0].as<unsigned int>();
        //     pano.resolution[1] = node["resolution"][1].as<unsigned int>();
        // }

        // if(node["wavelength"])
        // {
        //     pano.wavelength[0] = node["wavelength"][0].as<double>();
        //     pano.wavelength[1] = node["wavelength"][0].as<double>();
        // }

        return true;
    }
};

} // namespace YAML

#endif // LVR2_IO_YAML_HYPERSPECTRALCAMERAMETA_IO_HPP
