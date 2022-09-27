
#ifndef LVR2_IO_YAML_SCANPOSMETA_IO_HPP
#define LVR2_IO_YAML_SCANPOSMETA_IO_HPP

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/util/YAMLUtil.hpp"
#include "lvr2/io/YAML.hpp"

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
        Node node = scanPos.metadata;
        
        node["entity"] = lvr2::ScanPosition::entity;
        node["type"] = lvr2::ScanPosition::type;
        node["pose_estimation"] = scanPos.poseEstimation;
        node["transformation"] = scanPos.transformation;
        node["timestamp"] = scanPos.timestamp;
        node["original_name"] = scanPos.original_name;

        if(scanPos.boundingBox)
        {
            node["aabb"] = *scanPos.boundingBox;
        }

        return node;
    }

    static bool decode(const Node& node, lvr2::ScanPosition& scanPos) 
    {
        //Auskommentiert von der RDBX Gruppe
        // Check if 'entity' and 'type' Tags are valid
        // maybe checking for both is redundant because they are the same
        // but maybe this changes in the future, so just leave it like this
    /*    if (!YAML_UTIL::ValidateEntityAndType(node,
            "scan_position",
            lvr2::ScanPosition::entity,
            lvr2::ScanPosition::type))
        {
            return false;
        }
*/

        if(node["original_name"])
        {
            try {
                scanPos.original_name = node["original_name"].as<double>();
            } catch(const YAML::TypedBadConversion<double>& ex) {
                std::cerr << "[YAML - ScanPosition - decode] ERROR: Could not decode 'orginal_name': "
                          << node["original_name"] << " as double" << std::endl;
                return false;
            }
        } else {
            // TODO: was soll hier gespeichert werden
            scanPos.original_name = -1;
        }



        if(node["pose_estimation"])
        {
            try {
                scanPos.poseEstimation = node["pose_estimation"].as<lvr2::Transformd>();
            } catch(const YAML::TypedBadConversion<lvr2::Transformd>& ex) {
                std::cerr << "[YAML - ScanPosition - decode] ERROR: Could not decode 'pose_estimation': " 
                    << node["pose_estimation"] << " as Transformd" << std::endl; 
                return false;
            }
        } else {
            scanPos.poseEstimation = lvr2::Transformd::Identity();
        }
        
        if(node["transformation"])
        {
            try {
                scanPos.transformation = node["transformation"].as<lvr2::Transformd>();
            } catch(const YAML::TypedBadConversion<lvr2::Transformd>& ex) {
                std::cerr << "[YAML - ScanPosition - decode] ERROR: Could not decode 'transformation': " 
                    << node["transformation"] << " as Transformd" << std::endl; 
                return false;
            }
        } else {
            scanPos.transformation = lvr2::Transformd::Identity();
        }
        
        if(node["timestamp"])
        {
            try {
                scanPos.timestamp = node["timestamp"].as<double>();
            } catch(const YAML::TypedBadConversion<double>& ex) {
                std::cerr << "[YAML - ScanPosition - decode] ERROR: Could not decode 'timestamp': " 
                    << node["timestamp"] << " as double" << std::endl; 
                return false;
            }
        } else {
            scanPos.timestamp = -1.0;
        }

        if(node["aabb"])
        {
            try {
                scanPos.boundingBox = node["aabb"].as<lvr2::BoundingBox<lvr2::BaseVector<float> > >();
            } catch(const YAML::TypedBadConversion<lvr2::BoundingBox<lvr2::BaseVector<float> > >& ex) {
                std::cerr << "[YAML - ScanPosition - decode] ERROR: Could not decode 'aabb': " 
                    << node["aabb"] << " as BoundingBox" << std::endl; 
                return false;
            }
        }

        scanPos.metadata= node;
        //TODO: Echten Namen finden eventuell bessere Ort finden



        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_SCANPOSMETA_IO_HPP

