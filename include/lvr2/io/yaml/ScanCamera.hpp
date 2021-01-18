
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
        node["sensor_type"] = lvr2::ScanCamera::sensorType;
        node["sensor_name"] = scanCam.sensorName;
        node["camera_model"] = "pinhole";
        node["distortion_model"] = scanCam.camera.distortionModel;
        node["resolution"] = Load("[]");
        node["resolution"].push_back(scanCam.camera.width);
        node["resolution"].push_back(scanCam.camera.height);
        
        Node pinholeNode;
        pinholeNode["cx"] = scanCam.camera.cx;
        pinholeNode["cy"] = scanCam.camera.cy;
        pinholeNode["fx"] = scanCam.camera.fx;
        pinholeNode["fy"] = scanCam.camera.fy;
        node["pinhole"] = pinholeNode;

        if(scanCam.camera.distortionModel == "opencv")
        {
            Node distortion = Load("[]");
            for(size_t i = 0; i < scanCam.camera.k.size(); i++)
            {
                distortion.push_back(scanCam.camera.k[i]);
            }
            node["distortion_coefficients"] = distortion;
        } else {
            // unkown distortion model
        }

        return node;
    }

    static bool decode(const Node& node, lvr2::ScanCamera& scanCam) 
    {
        // Check if we are reading camera information
        if(node["sensor_type"].as<std::string>() != lvr2::ScanCamera::sensorType)
        {
            return false;
        }

        if(node["sensor_name"])
        {
            scanCam.sensorName = node["sensor_name"].as<std::string>();
        }
        else
        {
            scanCam.sensorName = "noname";
        }

        if(node["resolution"] && node["resolution"].size() == 2)
        {
            scanCam.camera.width = node["resolution"][0].as<unsigned>();
            scanCam.camera.height = node["resolution"][1].as<unsigned>();
        }

        if(node["camera_model"])
        {
            std::string camera_model = node["camera_model"].as<std::string>();
            if(camera_model == "pinhole")
            {
                if(node["pinhole"])
                {
                    // load pinhole parameters
                    Node pinholeNode = node["pinhole"];
                    scanCam.camera.cx = pinholeNode["cx"].as<double>();
                    scanCam.camera.cy = pinholeNode["cy"].as<double>();
                    scanCam.camera.fx = pinholeNode["fx"].as<double>();
                    scanCam.camera.fy = pinholeNode["fy"].as<double>();
                }
                
            } else {
                std::cout << "Camera model unknown" << std::endl;
            }
        }        

        // Check if we have distortion data in OpenCV format

        scanCam.camera.distortionModel = node["distortion_model"].as<std::string>();

        if(scanCam.camera.distortionModel == "opencv")
        {
            Node distortionNode = node["distortion_coefficients"];
            scanCam.camera.k.clear();
            if(distortionNode)
            {
                YAML::const_iterator it = distortionNode.begin();
                YAML::const_iterator it_end = distortionNode.end();

                for(; it != it_end; it++)
                {
                    scanCam.camera.k.push_back(it->as<double>());
                }
            }
        } else {
            std::cout << "Distortion model unknown" << std::endl;
        }

        
       

        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_PINHOLECAMERAMODEL_IO_HPP

