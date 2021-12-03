
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
            "spectral_camera", 
            lvr2::HyperspectralCamera::entity, 
            lvr2::HyperspectralCamera::type))
        {
            return false;
        }

        if(node["transformation"])
        {
            camera.transformation = node["transformation"].as<decltype(camera.transformation)>();
        }

        if(node["name"])
        {
            camera.name = node["name"].as<decltype(camera.name)>();
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
        node["frames_resolution"] = Load("[]");
        node["frames_resolution"].push_back(pano.framesResolution[0]);
        node["frames_resolution"].push_back(pano.framesResolution[1]);
        node["frames_resolution"].push_back(pano.framesResolution[2]);

        node["band_axis"] = pano.bandAxis;
        node["frame_axis"] = pano.frameAxis;
        node["data_type"] = pano.dataType;

        node["panorama_resolution"] = Load("[]");
        node["panorama_resolution"].push_back(pano.panoramaResolution[0]);
        node["panorama_resolution"].push_back(pano.panoramaResolution[1]);
        node["panorama_resolution"].push_back(pano.panoramaResolution[2]);

        if(pano.model)
        {
            node["model"] = *pano.model;
        }

        node["preview_type"] = pano.previewType;

        return node;
    }

    static bool decode(const Node& node, lvr2::HyperspectralPanorama& pano)
    {
        // Check if 'entity' and 'type' Tags are valid
        if (!YAML_UTIL::ValidateEntityAndType(node, 
            "spectral_panorama", 
            lvr2::HyperspectralPanorama::entity, 
            lvr2::HyperspectralPanorama::type))
        {
            return false;
        }

        if(node["transformation"])
        {
            pano.transformation = node["transformation"].as<lvr2::Transformd>();
        }

        if(node["frames_resolution"])
        {
            pano.framesResolution[0] = node["frames_resolution"][0].as<unsigned int>();
            pano.framesResolution[1] = node["frames_resolution"][1].as<unsigned int>();
            pano.framesResolution[2] = node["frames_resolution"][1].as<unsigned int>();
        }

        if(node["band_axis"])
        {
            pano.bandAxis = node["band_axis"].as<unsigned int>();
        }

        if(node["frame_axis"])
        {
            pano.frameAxis = node["frame_axis"].as<unsigned int>();
        }

        if(node["data_type"])
        {
            pano.dataType = node["data_type"].as<std::string>();
        }

        if(node["panorama_resolution"])
        {
            pano.panoramaResolution[0] = node["panorama_resolution"][0].as<unsigned int>();
            pano.panoramaResolution[1] = node["panorama_resolution"][1].as<unsigned int>();
            pano.panoramaResolution[2] = node["panorama_resolution"][1].as<unsigned int>();
        }

        if(node["model"])
        {
            pano.model = node["model"].as<lvr2::CylindricalModel>();
        }

        if(node["preview_type"])
        {
            pano.previewType = node["preview_type"].as<std::string>();
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
