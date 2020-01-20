
#ifndef LVR2_IO_YAML_SCANMETA_IO_HPP
#define LVR2_IO_YAML_SCANMETA_IO_HPP

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
struct convert<lvr2::Scan> 
{

    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::Scan& scan) {
        Node node;
        
        node["start_time"]  = scan.m_startTime;
        node["end_time"] = scan.m_endTime;

        node["pose_estimate"] = scan.m_poseEstimation;
        node["registration"] = scan.m_registration;

        Node config;
        config["theta"].push_back(scan.m_thetaMin);
        config["theta"].push_back(scan.m_thetaMax);

        config["phi"].push_back(scan.m_phiMin);
        config["phi"].push_back(scan.m_phiMax);

        config["v_res"] = scan.m_vResolution;
        config["h_res"] = scan.m_hResolution;

        config["num_points"] = scan.m_numPoints;
        node["config"] = config;

        return node;
    }

    static bool decode(const Node& node, lvr2::Scan& scan) {
        
        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_SCANMETA_IO_HPP

