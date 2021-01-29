
#ifndef LVR2_IO_YAML_CAMERA_HPP
#define LVR2_IO_YAML_CAMERA_HPP

#include <sstream>

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
template<>
struct convert<lvr2::Camera> 
{

    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::Camera& camera) {
        
        Node node;
        node["type"] = lvr2::Camera::type;
        node["kind"] = lvr2::Camera::kind;
        
        node["name"] = camera.name;
        
        node["transformation"] = camera.transformation;
        
        node["camera_model"] = "pinhole";
        node["distortion_model"] = camera.model.distortionModel;
        node["resolution"] = Load("[]");
        node["resolution"].push_back(camera.model.width);
        node["resolution"].push_back(camera.model.height);
        
        Node pinholeNode;
        pinholeNode["c"] = Load("[]");
        pinholeNode["c"].push_back(camera.model.cx);
        pinholeNode["c"].push_back(camera.model.cy);
        pinholeNode["f"] = Load("[]");
        pinholeNode["f"].push_back(camera.model.fx);
        pinholeNode["f"].push_back(camera.model.fy);
        node["pinhole"] = pinholeNode;

        if(camera.model.distortionModel == "opencv")
        {
            Node distortion = Load("[]");
            for(size_t i = 0; i < camera.model.k.size(); i++)
            {
                distortion.push_back(camera.model.k[i]);
            }
            node["distortion_coefficients"] = distortion;
        } else {
            // unkown distortion model
        }

        return node;
    }

    static bool decode(const Node& node, lvr2::Camera& camera) 
    {
        // Check if we are reading camera information
        if(!node["type"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<Camera> - decode] "
                     << "Camera meta has no key 'type'" << std::endl; 
            return false;
        }

        if(node["type"].as<std::string>() != lvr2::Camera::type)
        {
            std::cout << lvr2::timestamp << "[YAML::convert<Camera> - decode] " 
                        << "Nodes type '" << node["type"].as<std::string>()
                        << "' is not '" <<  lvr2::Camera::type << "'" << std::endl; 
            return false;
        }

        if(!node["kind"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<Camera> - decode] "
                     << "Sensor has no key 'kind'" << std::endl;
            return false; 
        }

        if(node["kind"].as<std::string>() != lvr2::Camera::kind)
        {
            std::cout << lvr2::timestamp << "[YAML::convert<Camera> - decode] " 
                        << "Nodes kind '" << node["kind"].as<std::string>()
                        << "' is not '" <<  lvr2::Camera::kind << "'" << std::endl; 
            return false;
        }

        if(node["name"])
        {
            camera.name = node["name"].as<std::string>();
        }
        else
        {
            camera.name = "";
        }

        if(node["transformation"])
        {
            camera.transformation = node["transformation"].as<lvr2::Transformd>();
        }

        if(node["resolution"] && node["resolution"].size() == 2)
        {
            camera.model.width = node["resolution"][0].as<unsigned>();
            camera.model.height = node["resolution"][1].as<unsigned>();
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
                    camera.model.cx = pinholeNode["c"][0].as<double>();
                    camera.model.cy = pinholeNode["c"][1].as<double>();
                    camera.model.fx = pinholeNode["f"][0].as<double>();
                    camera.model.fy = pinholeNode["f"][1].as<double>();
                }
                
            } else {
                std::cout << "Camera model unknown" << std::endl;
            }
        }        

        // Check if we have distortion data in OpenCV format

        camera.model.distortionModel = node["distortion_model"].as<std::string>();

        if(camera.model.distortionModel == "opencv")
        {
            Node distortionNode = node["distortion_coefficients"];
            camera.model.k.clear();
            if(distortionNode)
            {
                YAML::const_iterator it = distortionNode.begin();
                YAML::const_iterator it_end = distortionNode.end();

                for(; it != it_end; it++)
                {
                    camera.model.k.push_back(it->as<double>());
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

