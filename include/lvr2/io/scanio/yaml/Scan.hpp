
#ifndef LVR2_IO_YAML_SCANMETA_IO_HPP
#define LVR2_IO_YAML_SCANMETA_IO_HPP

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "Matrix.hpp"
#include "AABB.hpp"

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
            scan.startTime = node["start_time"].as<double>();
        } else {
            scan.startTime = -1.0;
        }

        if(node["end_time"])
        {
            scan.endTime = node["end_time"].as<double>();
        } else {
            scan.endTime = -1.0;
        }

        if(node["num_points"])
        {
            scan.numPoints = node["num_points"].as<unsigned int>();
        }
        
        if(node["pose_estimation"])
        {
            scan.poseEstimation = node["pose_estimation"].as<lvr2::Transformd>();
        } else {
            scan.poseEstimation = lvr2::Transformd::Identity();
        }

        if(node["transformation"])
        {
            scan.transformation = node["transformation"].as<lvr2::Transformd>();
        } else {
            scan.transformation = lvr2::Transformd::Identity();
        }

        if(node["model"])
        {
            scan.model = node["model"].as<lvr2::SphericalModel>();
        }
        
        if(node["aabb"])
        {
            scan.boundingBox = node["aabb"].as<lvr2::BoundingBox<lvr2::BaseVector<float> > >();
        }

        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_SCANMETA_IO_HPP

