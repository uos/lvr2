
#ifndef LVR2_IO_YAML_HYPERSPECTRALCAMERAMETA_IO_HPP
#define LVR2_IO_YAML_HYPERSPECTRALCAMERAMETA_IO_HPP

#include <yaml-cpp/yaml.h>

#include "lvr2/io/YAML.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/util/YAMLUtil.hpp"

using lvr2::timestamp;


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
            try 
            {
                camera.transformation = node["transformation"].as<decltype(camera.transformation)>();
            } 
            catch(const YAML::TypedBadConversion<lvr2::Transformd>& ex) 
            {
                std::cout << timestamp << "[YAML - HyperspectralCamera - decode] ERROR: Could not decode 'transformation': "
                    << node["transformation"] << " as Transformd" << std::endl; 
                return false;
            }
        }

        if(node["name"])
        {
            try 
            {
                camera.name = node["name"].as<decltype(camera.name)>();
            } 
            catch(const YAML::TypedBadConversion<std::string>& ex) 
            {
                std::cout << timestamp << "[YAML - HyperspectralCamera - decode] ERROR: Could not decode 'name': "
                    << node["name"] << " as string" << std::endl; 
                return false;
            }
        }

        if(node["model"])
        {
            try 
            {
                camera.model= node["model"].as<decltype(camera.model)>();
            } 
            catch(const YAML::TypedBadConversion<decltype(camera.model)>& ex) 
            {
                std::cout << timestamp << "[YAML - HyperspectralCamera - decode] ERROR: Could not decode 'model': "
                    << node["model"] << " as CameraModel" << std::endl;
                return false;
            }
        } 
        else 
        {
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
            "spectral_panorama_channel", 
            lvr2::HyperspectralPanoramaChannel::entity, 
            lvr2::HyperspectralPanoramaChannel::type))
        {
            return false;
        }

        return true;
    }
};

} // namespace YAML

#endif // LVR2_IO_YAML_HYPERSPECTRALCAMERAMETA_IO_HPP
