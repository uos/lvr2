
#ifndef LVR2_IO_YAML_SCANPOSMETA_IO_HPP
#define LVR2_IO_YAML_SCANPOSMETA_IO_HPP

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "MatrixIO.hpp"

namespace YAML {  

/**
 * YAML-CPPs convert specialization
 * 
 * example: 
 */

// WRITE SCAN PARTIALLY
template <>
struct convert<lvr2::ScanPosition> 
{

    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::ScanPosition& scanPos) {
        Node node;
        
        node["sensor_type"] = scanPos.sensorType;
        node["latitude"] = scanPos.latitude;
        node["longitude"] = scanPos.longitude;
        node["poseEstimate"] = scanPos.poseEstimate;
        node["registration"] = scanPos.registration;
        node["timestamp"] = scanPos.timestamp;

        return node;
    }

    static bool decode(const Node& node, lvr2::ScanPosition& scanPos) {
        
        if(node["sensor_type"].as<std::string>() != scanPos.sensorType)
        {
            return false;
        }

        scanPos.latitude = node["latitude"].as<double>();
        scanPos.longitude = node["longitude"].as<double>();
        
        scanPos.poseEstimate = node["poseEstimate"].as<lvr2::Transformd>();
        scanPos.registration = node["registration"].as<lvr2::Transformd>();

        scanPos.timestamp = node["timestamp"].as<double>();

        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_SCANPOSMETA_IO_HPP

