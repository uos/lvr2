
#ifndef LVR2_IO_YAML_HYPERSPECTRALPANORAMACHANNELMETA_IO_HPP
#define LVR2_IO_YAML_HYPERSPECTRALPANORAMACHANNELMETA_IO_HPP

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
struct convert<lvr2::HyperspectralPanoramaChannel>
{

    /**
     * Encode Eigen matrix to yaml.
     */
    static Node encode(const lvr2::HyperspectralPanoramaChannel& channel)
    {
        Node node;

        node["sensor_type"] = lvr2::HyperspectralPanoramaChannel::sensorType;

        node["timestamp"] = channel.timestamp;

        return node;
    }

    static bool decode(const Node& node, lvr2::HyperspectralPanoramaChannel& channel)
    {

        if (node["sensor_type"].as<std::string>() != lvr2::HyperspectralPanoramaChannel::sensorType)
        {
            return false;
        }

        channel.timestamp = node["timestamp"].as<double>();

        return true;
    }
};

} // namespace YAML

#endif // LVR2_IO_YAML_HYPERSPECTRALPANORAMACHANNELMETA_IO_HPP
