
#ifndef LVR2_IO_YAML_PINHOLECAMERAMODEL_IO_HPP
#define LVR2_IO_YAML_PINHOLECAMERAMODEL_IO_HPP

#include <sstream>

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "MatrixIO.hpp"
#include "lvr2/registration/CameraModels.hpp"

namespace YAML {

/**
 * YAML-CPPs convert specialization
 * 
 * example: 
 */

// WRITE SCAN PARTIALLY
template<>
struct convert<lvr2::ScanCamera> 
{

    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::ScanCamera& scanCam) {
        
        Node node;
        node["sensor_type"] = scanCam.sensorType;
        node["sensor_name"] = scanCam.sensorName;
        node["camera_model"] = "pinhole";
        node["distortion_model"] = scanCam.camera.distortionModel;
        node["resolution"].push_back(scanCam.camera.width);
        node["resolution"].push_back(scanCam.camera.height);
        
        Node pinholeNode;
        pinholeNode["cx"] = scanCam.camera.cx;
        pinholeNode["cy"] = scanCam.camera.cy;
        pinholeNode["fx"] = scanCam.camera.fx;
        pinholeNode["fy"] = scanCam.camera.fy;
        node["pinhole"] = pinholeNode;

        Node distortion;
        for(size_t i = 0; i < scanCam.camera.k.size(); i++)
        {
            std::stringstream s;
            s << "k" << i;
            distortion[s.str()] = scanCam.camera.k[i];
        }
        node["distortion"] = distortion;

        return node;
    }

    static bool decode(const Node& node, lvr2::ScanCamera& scanCam) 
    {
        // Check if we are reading camera information
        if(node["sensor_type"].as<std::string>() != "camera")
        {
            return false;
        }
        

        // Check if we have distortion data in OpenCV format
        if(node["distortion_model"].as<std::string>() != "opencv")
        {
            return false;
        }

        // Get fields
        if(node["sensor_type"])
        {
            scanCam.sensorType = node["sensor_type"].as<std::string>();
        }
        else
        {
            scanCam.sensorType = "unknown";
        }

        if(node["sensor_name"])
        {
            scanCam.sensorType = node["sensor_type"].as<std::string>();
        }
        else
        {
            scanCam.sensorType = "noname";
        }

        if(node["resolution"] && node["resolution"].size() == 2)
        {
            scanCam.camera.width = node["resolution"][0].as<int>();
            scanCam.camera.height = node["resolution"][1].as<int>();
        }
       
        Node pinholeNode = node["pinhole"];
        if(pinholeNode)
        {
            if(pinholeNode["fx"])
            {
                scanCam.camera.fx = pinholeNode["fx"].as<double>();
            }

            if(pinholeNode["fy"])
            {
                scanCam.camera.fx = pinholeNode["fy"].as<double>();
            }

            if(pinholeNode["cx"])
            {
                scanCam.camera.cx = pinholeNode["cx"].as<double>();
            }

            if(pinholeNode["cy"])
            {
                scanCam.camera.cx = pinholeNode["cy"].as<double>();
            }
        }

        Node distortionNode = node["distortion"];
        scanCam.camera.k.clear();
        if(distortionNode)
        {
            for(size_t i = 0; i < distortionNode.size(); i++)
            {
                scanCam.camera.k.push_back(distortionNode[i].as<double>());
            }
        }

        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_PINHOLECAMERAMODEL_IO_HPP

