
#ifndef LVR2_IO_YAML_SCANPROJECTMETA_IO_HPP
#define LVR2_IO_YAML_SCANPROJECTMETA_IO_HPP

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
struct convert<lvr2::ScanProject> 
{

    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::ScanProject& scanProj) {
        Node node;
        
        node["sensor_type"] = lvr2::ScanProject::sensorType;

        return node;
    }

    static bool decode(const Node& node, lvr2::ScanProject& scanProj) {
        
        if(node["sensor_type"].as<std::string>() != lvr2::ScanProject::sensorType)
        {
            return false;
        }



        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_SCANPROJECTMETA_IO_HPP

