
#ifndef LVR2_IO_YAML_SCANMETA_IO_HPP
#define LVR2_IO_YAML_SCANMETA_IO_HPP

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
struct convert<lvr2::Scan> 
{

    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::Scan& scan) {
        Node node;
        
        node["sensor_type"] = lvr2::Scan::sensorType;

        node["start_time"]  = scan.startTime;
        node["end_time"] = scan.endTime;

        node["pose_estimate"] = scan.poseEstimation;
        node["registration"] = scan.registration;

        Node config;
        config["theta"] = Load("[]");
        config["theta"].push_back(scan.thetaMin);
        config["theta"].push_back(scan.thetaMax);

        config["phi"] = Load("[]");
        config["phi"].push_back(scan.phiMin);
        config["phi"].push_back(scan.phiMax);

        config["v_res"] = scan.vResolution;
        config["h_res"] = scan.hResolution;

        config["num_points"] = scan.numPoints;
        node["config"] = config;

        return node;
    }

    static bool decode(const Node& node, lvr2::Scan& scan) {
        
        if(node["sensor_type"].as<std::string>() != lvr2::Scan::sensorType) 
        {
            return false;
        }

        scan.startTime = node["start_time"].as<double>();
        scan.endTime = node["end_time"].as<double>();
        scan.poseEstimation = node["pose_estimate"].as<lvr2::Transformd>();
        scan.registration = node["registration"].as<lvr2::Transformd>();
        
        const Node& config = node["config"];

        
        scan.thetaMin = config["theta"][0].as<double>();
        scan.thetaMax = config["theta"][1].as<double>();

        scan.phiMin = config["phi"][0].as<double>();
        scan.phiMax = config["phi"][1].as<double>();


        scan.vResolution = config["v_res"].as<double>();
        scan.hResolution = config["h_res"].as<double>();

        scan.numPoints = config["num_points"].as<size_t>();


        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_SCANMETA_IO_HPP

