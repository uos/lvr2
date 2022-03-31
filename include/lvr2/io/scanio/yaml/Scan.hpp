
#ifndef LVR2_IO_YAML_SCANMETA_IO_HPP
#define LVR2_IO_YAML_SCANMETA_IO_HPP


#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "lvr2/io/YAML.hpp"

namespace YAML {  

/**
 * YAML-CPPs convert specialization
 * 
 * example: 
 */

// WRITE SCAN PARTIALLY
template <>
struct convert<lvr2::Scan> 
{

    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::Scan& scan) {

        // std::cout << "Encode Scan"

        Node node;
        node["entity"] = lvr2::Scan::entity;
        node["type"] = lvr2::Scan::type;

        node["start_time"]  = scan.startTime;
        node["end_time"] = scan.endTime;
        node["num_points"] = scan.numPoints;

        node["pose_estimation"] = scan.poseEstimation;
        node["transformation"] = scan.transformation;

        if(scan.boundingBox)
        {
            node["aabb"] = *scan.boundingBox;
        }

        if(scan.model)
        {
            node["model"] = *scan.model;
        }

        if(scan.points)
        {
            node["channels"] = Load("[]");
            for(auto elem : *scan.points)
            {
                node["channels"].push_back(elem.first);
            }
        }

        return node;
    }

    static bool decode(const Node& node, lvr2::Scan& scan)
    {
        // Check if 'entity' and 'type' Tags are valid
        if (!YAML_UTIL::ValidateEntityAndType(node, 
            "scan", 
            lvr2::Scan::entity, 
            lvr2::Scan::type))
        {
            return false;
        }

        if(node["start_time"])
        {
            try {
                scan.startTime = node["start_time"].as<double>();
            } catch(const YAML::TypedBadConversion<double>& ex) {
                std::cerr << "[YAML - Scan - decode] ERROR: Could not convert 'start_time': " << node["start_time"] << " to double" << std::endl; 
                return false;
            }
        } else {
            scan.startTime = -1.0;
        }

        if(node["end_time"])
        {
            try {
                scan.endTime = node["end_time"].as<double>();
            } catch(const YAML::TypedBadConversion<double>& ex) {
                std::cerr << "[YAML - Scan - decode] ERROR: Could not convert 'end_time': " << node["end_time"] << " to double" << std::endl; 
                return false;
            }
        } else {
            scan.endTime = -1.0;
        }

        if(node["num_points"])
        {
            try {
                scan.numPoints = node["num_points"].as<unsigned int>();
            } catch(const YAML::TypedBadConversion<unsigned int>& ex) {
                std::cerr << "[YAML - Scan - decode] ERROR: Could not convert 'num_points': " << node["num_points"] << " to unsigned int" << std::endl; 
                return false;
            }
        }
        
        if(node["pose_estimation"])
        {
            try {
                scan.poseEstimation = node["pose_estimation"].as<lvr2::Transformd>();
            } catch(const YAML::TypedBadConversion<lvr2::Transformd>& ex) {
                std::cerr << "[YAML - Scan - decode] ERROR: Could not convert 'pose_estimation': " << node["pose_estimation"] << " to Transformd" << std::endl; 
                return false;
            }
        } else {
            scan.poseEstimation = lvr2::Transformd::Identity();
        }

        if(node["transformation"])
        {
            try {
                scan.transformation = node["transformation"].as<lvr2::Transformd>();
            } catch(const YAML::TypedBadConversion<lvr2::Transformd>& ex) {
                std::cerr << "[YAML - Scan - decode] ERROR: Could not convert 'transformation': " << node["transformation"] << " to Transformd" << std::endl; 
                return false;
            }
        } else {
            scan.transformation = lvr2::Transformd::Identity();
        }

        if(node["model"])
        {
            try {
                scan.model = node["model"].as<lvr2::SphericalModel>();
            } catch(const YAML::TypedBadConversion<lvr2::SphericalModel>& ex) {
                std::cerr << "[YAML - Scan - decode] ERROR: Could not convert 'model': " << node["model"] << " to SphericalModel" << std::endl; 
                return false;
            }
        }
        
        if(node["aabb"])
        {
            try {
                scan.boundingBox = node["aabb"].as<lvr2::BoundingBox<lvr2::BaseVector<float> > >();
            } catch(const YAML::TypedBadConversion<lvr2::BoundingBox<lvr2::BaseVector<float> > >& ex) {
                std::cerr << "[YAML - Scan - decode] ERROR: Could not convert 'aabb': " << node["aabb"] << " to BoundingBox" << std::endl; 
                return false;
            }
        }

        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_SCANMETA_IO_HPP

