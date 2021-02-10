
#ifndef LVR2_IO_YAML_HYPERSPECTRALCAMERAMETA_IO_HPP
#define LVR2_IO_YAML_HYPERSPECTRALCAMERAMETA_IO_HPP

#include <yaml-cpp/yaml.h>
#include "Matrix.hpp"
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

        node["transformation"] = camera.transformation;
        
        node["camera_model"] = "cylindric";



        if(camera.model)
        {
            Node model_node;

            model_node["focal_length"] = camera.model->focalLength;
            model_node["offset_angle"] = camera.model->offsetAngle;

            Node principal_node = Load("[]");
            principal_node.push_back(camera.model->principal(0));
            principal_node.push_back(camera.model->principal(1));
            principal_node.push_back(camera.model->principal(2));
            model_node["principal"] = principal_node;

            Node distortion_node = Load("[]");
            distortion_node.push_back(camera.model->distortion(0));
            distortion_node.push_back(camera.model->distortion(1));
            distortion_node.push_back(camera.model->distortion(2));
            model_node["distortion"] = principal_node;

            node["model"] = model_node;
        }

        return node;
    }

    static bool decode(const Node& node, lvr2::HyperspectralCamera& camera)
    {
        if (node["type"].as<std::string>() != lvr2::HyperspectralCamera::type)
        {
            // different hierarchy level
            return false;
        }

        if(node["kind"].as<std::string>() != lvr2::HyperspectralCamera::kind)
        {
            // different sensor type
            return false;
        }

        return true;
    }
};

} // namespace YAML

#endif // LVR2_IO_YAML_HYPERSPECTRALCAMERAMETA_IO_HPP
