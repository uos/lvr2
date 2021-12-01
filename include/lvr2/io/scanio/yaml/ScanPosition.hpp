
#ifndef LVR2_IO_YAML_SCANPOSMETA_IO_HPP
#define LVR2_IO_YAML_SCANPOSMETA_IO_HPP

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "Matrix.hpp"
#include "lvr2/io/scanio/yaml/Util.hpp"

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
        
        node["entity"] = lvr2::ScanPosition::entity;
        node["type"] = lvr2::ScanPosition::type;
        node["pose_estimation"] = scanPos.poseEstimation;
        node["transformation"] = scanPos.transformation;
        node["timestamp"] = scanPos.timestamp;
        return node;
    }

    static bool decode(const Node& node, lvr2::ScanPosition& scanPos) 
    {
        // Check if 'entity' and 'type' Tags are valid
        // maybe checking for both is redundant because they are the same
        // but maybe this changes in the future, so just leave it like this
        if (!YAML_UTIL::ValidateEntityAndType(node, 
            "scan_position", 
            lvr2::ScanPosition::entity, 
            lvr2::ScanPosition::type))
        {
            return false;
        }
        
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

