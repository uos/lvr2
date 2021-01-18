
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
        
        node["sensor_type"] = lvr2::ScanPosition::sensorType;
        node["latitude"] = scanPos.latitude;
        node["longitude"] = scanPos.longitude;
        node["altitude"] = scanPos.altitude;
        node["pose_estimate"] = scanPos.pose_estimate;
        node["registration"] = scanPos.registration;
        node["timestamp"] = scanPos.timestamp;

        return node;
    }

    static bool decode(const Node& node, lvr2::ScanPosition& scanPos) 
    {
        if(!node["sensor_type"])
        {
            std::cout << "[YAML::convert<ScanPosition> - decode] 'sensor_type' Tag not found." << std::endl;
            return false;
        }    

        if(node["sensor_type"].as<std::string>() != lvr2::ScanPosition::sensorType)
        {
            std::cout << "[YAML::convert<ScanPosition> - decode] Try to load " << node["sensor_type"].as<std::string>() << " as " << lvr2::ScanPosition::sensorType << std::endl;
            return false;
        }
        
        if(node["latitude"])
        {
            scanPos.latitude = node["latitude"].as<double>();
        }
        
        if(node["longitude"])
        {
            scanPos.longitude = node["longitude"].as<double>();
        }
        
        if(node["altitude"])
        {
            scanPos.altitude = node["altitude"].as<double>();
        }
        
        if(node["pose_estimate"])
        {
            scanPos.pose_estimate = node["pose_estimate"].as<lvr2::Transformd>();
        } else {
            scanPos.pose_estimate = lvr2::Transformd::Identity();
        }
        
        if(node["registration"])
        {
            scanPos.registration = node["registration"].as<lvr2::Transformd>();
        } else {
            scanPos.registration = lvr2::Transformd::Identity();
        }
        
        if(node["timestamp"])
        {
            scanPos.timestamp = node["timestamp"].as<double>();
        } else {
            scanPos.timestamp = -1.0;
        }

        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_SCANPOSMETA_IO_HPP

