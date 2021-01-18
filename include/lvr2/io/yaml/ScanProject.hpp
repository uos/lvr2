
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
        try
        {
            if(node["sensor_type"].as<std::string>() != lvr2::ScanProject::sensorType)
            {
                return false;
            }

            scanProj.pose = node["pose_estimate"].as<lvr2::Transformd>();
            scanProj.coordinateSystem = node["coordinate_system"].as<std::string>();
            scanProj.sensorName = node["sensor_name"].as<std::string>();
        }
        catch(YAML::BadSubscript& e)
        {
            std::cout << e.what() << std::endl;
            return false;
        }
    
        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_SCANPROJECTMETA_IO_HPP

