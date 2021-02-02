
#ifndef LVR2_IO_YAML_SCANPROJECTMETA_IO_HPP
#define LVR2_IO_YAML_SCANPROJECTMETA_IO_HPP

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/Timestamp.hpp"

#include "MatrixIO.hpp"

namespace YAML {  

/**
 * YAML-CPPs convert specialization
 * 
 * example: 
 */

// WRITE SCAN PARTIALLY
template <>
struct convert<lvr2::ScanProject> 
{

    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::ScanProject& scanProj) {
        Node node;
        
        node["sensor_type"] = lvr2::ScanProject::sensorType;
        node["coordinate_system"] = scanProj.coordinateSystem;
        node["pose_estimate"] = scanProj.pose;
        node["sensor_name"] = scanProj.sensorName;

        return node;
    }

    static bool decode(const Node& node, lvr2::ScanProject& scanProj) 
    {
        if(!node["sensor_type"])
        {
            std::cout << "[YAML::convert<ScanProject> - decode] 'sensor_type' Tag not found." << std::endl;
            return false;
        }

        if(node["sensor_type"].as<std::string>() != lvr2::ScanProject::sensorType)
        {
            std::cout << "[YAML::convert<ScanProject> - decode] Try to load " << node["sensor_type"].as<std::string>() << " as " << lvr2::ScanProject::sensorType << std::endl;
            return false;
        }

        if(node["pose_estimate"])
        {
            scanProj.pose = node["pose_estimate"].as<lvr2::Transformd>();
        }  else {
            scanProj.pose  = lvr2::Transformd::Identity();
        }
      
        if(node["coordinate_system"])
        {
            scanProj.coordinateSystem = node["coordinate_system"].as<std::string>();
        }
              
        if(node["sensor_name"])
        {
            scanProj.sensorName = node["sensor_name"].as<std::string>();
        }
    
        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_SCANPROJECTMETA_IO_HPP

