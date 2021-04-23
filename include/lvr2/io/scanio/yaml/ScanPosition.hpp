
#ifndef LVR2_IO_YAML_SCANPOSMETA_IO_HPP
#define LVR2_IO_YAML_SCANPOSMETA_IO_HPP

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "Matrix.hpp"

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
        
        node["type"] = lvr2::ScanPosition::type;
        // node["entity"] = lvr2::ScanPosition::type;
        // node["kind"] = boost::typeindex::type_id<lvr2::ScanPosition>().pretty_name();
        node["pose_estimation"] = scanPos.poseEstimation;
        node["transformation"] = scanPos.transformation;
        node["timestamp"] = scanPos.timestamp;
        return node;
    }

    static bool decode(const Node& node, lvr2::ScanPosition& scanPos) 
    {
        // if(!node["type"])
        // {
        //     std::cout << "[YAML::convert<ScanPosition> - decode] 'type' Tag not found." << std::endl;
        //     return false;
        // }    

        // if(node["type"].as<std::string>() != lvr2::ScanPosition::type)
        // {
        //     std::cout << "[YAML::convert<ScanPosition> - decode] Try to load " << node["type"].as<std::string>() << " as " << lvr2::ScanPosition::type << std::endl;
        //     return false;
        // }
        
        if(node["pose_estimation"])
        {
            scanPos.poseEstimation = node["pose_estimation"].as<lvr2::Transformd>();
        } else {
            scanPos.poseEstimation = lvr2::Transformd::Identity();
        }
        
        if(node["transformation"])
        {
            scanPos.transformation = node["transformation"].as<lvr2::Transformd>();
        } else {
            scanPos.transformation = lvr2::Transformd::Identity();
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

