
#ifndef LVR2_IO_YAML_HYPERSPECTRALCAMERAMETA_IO_HPP
#define LVR2_IO_YAML_HYPERSPECTRALCAMERAMETA_IO_HPP

#include "MatrixIO.hpp"
#include "lvr2/types/ScanTypes.hpp"

#include <yaml-cpp/yaml.h>

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

        node["sensor_type"] = lvr2::HyperspectralCamera::sensorType;

        node["focalLength"] = camera.focalLength;
        node["offsetAngle"] = camera.offsetAngle;

        node["extrinsics"] = camera.extrinsics;
        node["extrinsicsEstimate"] = camera.extrinsicsEstimate;

        node["principal"] = Load("[]");
        node["principal"].push_back(camera.principal[0]);
        node["principal"].push_back(camera.principal[1]);
        node["principal"].push_back(camera.principal[2]);

        node["distortion"] = Load("[]");
        node["distortion"].push_back(camera.principal[0]);
        node["distortion"].push_back(camera.principal[1]);
        node["distortion"].push_back(camera.principal[2]);

        return node;
    }

    static bool decode(const Node& node, lvr2::HyperspectralCamera& camera)
    {

        if (node["sensor_type"].as<std::string>() != lvr2::HyperspectralCamera::sensorType)
        {
            return false;
        }

        camera.focalLength = node["focalLength"].as<double>();
        camera.offsetAngle = node["offsetAngle"].as<double>();
        camera.extrinsics = node["extrinsics"].as<lvr2::Extrinsicsd>();
        camera.extrinsicsEstimate = node["extrinsicsEstimate"].as<lvr2::Extrinsicsd>();

        camera.principal[0] = node["principal"][0].as<double>();
        camera.principal[1] = node["principal"][1].as<double>();
        camera.principal[2] = node["principal"][1].as<double>();

        camera.distortion[0] = node["distortion"][0].as<double>();
        camera.distortion[1] = node["distortion"][1].as<double>();
        camera.distortion[2] = node["distortion"][1].as<double>();

        return true;
    }
};

} // namespace YAML

#endif // LVR2_IO_YAML_HYPERSPECTRALCAMERAMETA_IO_HPP
