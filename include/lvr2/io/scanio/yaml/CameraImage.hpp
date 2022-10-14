
#ifndef LVR2_IO_YAML_CAMERAIMAGE_IO_HPP
#define LVR2_IO_YAML_CAMERAIMAGE_IO_HPP

#include <sstream>

#include <yaml-cpp/yaml.h>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "lvr2/util/YAMLUtil.hpp"
#include "lvr2/io/YAML.hpp"

using lvr2::timestamp;

namespace YAML {

template<>
struct convert<lvr2::CameraImage> 
{
    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::CameraImage& cameraImage) { 
        Node node = cameraImage.metadata;

        node["entity"] = lvr2::CameraImage::entity;
        node["type"] = lvr2::CameraImage::type;
        node["transformation"] = cameraImage.transformation;
        node["pose_estimation"] = cameraImage.extrinsicsEstimation;
        node["resolution"] = Load("[]");
        node["resolution"].push_back(cameraImage.image.cols); // X
        node["resolution"].push_back(cameraImage.image.rows); // Y
        node["resolution"].push_back(cameraImage.image.channels()); // Z: number of channels
        node["timestamp"] = cameraImage.timestamp;

        return node;
    }

    static bool decode(const Node& node, lvr2::CameraImage& scanImage) 
    {
        //skip this for now since we cant read the riegl scanprojects otherwise
       // Check if 'entity' and 'type' Tags are valid
//        if (!YAML_UTIL::ValidateEntityAndTypeSilent(node,
//            "camera_image",
//            lvr2::CameraImage::entity,
//            lvr2::CameraImage::type))
//        {
//            return false;
//        }

        Node parsed_node=node;
        // Get fields
        if(parsed_node["transformation"])
        {
            try 
            {
                scanImage.transformation = parsed_node["transformation"].as<lvr2::Transformd>();
            } catch(const YAML::TypedBadConversion<lvr2::Transformd>& ex) 
            {
                std::cout <<  timestamp <<  "[YAML - CameraImage - decode] ERROR: Could not decode 'transformation': " 
                    << parsed_node["transformation"] << " as Transformd" << std::endl;
                return false;
            }
            parsed_node.remove("transformation");
        }
        else
        {
            scanImage.transformation = lvr2::Transformd::Identity();
        }

        if(parsed_node["pose_estimation"])
        {
            // NAN check?
            try 
            {
                scanImage.extrinsicsEstimation = parsed_node["pose_estimation"].as<lvr2::Extrinsicsd>();
            } 
            catch(const YAML::TypedBadConversion<lvr2::Extrinsicsd>& ex) 
            {
                std::cout << timestamp << "[YAML - CameraImage - decode] ERROR: Could not decode 'pose_estimation': " 
                    << parsed_node["pose_estimation"] << " as Extrinsicsd" << std::endl;
                return false;
            }
            parsed_node.remove("pose_estimation");
        }
        else
        {
            scanImage.extrinsicsEstimation = lvr2::Extrinsicsd::Identity();
        }

        if(parsed_node["timestamp"])
        {
            try 
            {
                scanImage.timestamp = parsed_node["timestamp"].as<double>();
            } 
            catch(const YAML::TypedBadConversion<double>& ex) 
            {
                std::cout <<  timestamp <<  "[YAML - CameraImage - decode] ERROR: Could not decode 'timestamp': " 
                    << node["timestamp"] << " as double" << std::endl; 
                return false;
            }
            parsed_node.remove("timestamp");
        } 
        else 
        {
            // TODO: how to handle no timestamp?
            scanImage.timestamp = -1.0;
        }
        scanImage.metadata= parsed_node;

        return true;
    }

};


template<>
struct convert<lvr2::CameraImageGroup> 
{
    static Node encode(const lvr2::CameraImageGroup& cameraImageGroup) { 
        Node node;

        node["entity"] = lvr2::CameraImageGroup::entity;
        node["type"] = lvr2::CameraImageGroup::type;
        node["transformation"] = cameraImageGroup.transformation;

        return node;
    }

    static bool decode(const Node& node, lvr2::CameraImageGroup& cameraImageGroup) 
    {
       // Check if 'entity' and 'type' Tags are valid
        if (!YAML_UTIL::ValidateEntityAndTypeSilent(node, 
            "camera_images", 
            lvr2::CameraImageGroup::entity, 
            lvr2::CameraImageGroup::type))
        {
            return false;
        }
    
        // Get fields
        if(node["transformation"])
        {
            try {
                cameraImageGroup.transformation = node["transformation"].as<lvr2::Transformd>();
            } 
            catch(const YAML::TypedBadConversion<lvr2::Transformd>& ex) 
            {
                std::cout <<  timestamp <<  "[YAML - CameraImageGroup - decode] ERROR: Could not decode 'transformation': " 
                    << node["transformation"] << " as Transformd" << std::endl; 
                return false;
            }
        }
        else
        {
            cameraImageGroup.transformation = lvr2::Transformd::Identity();
        }
        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_CAMERAIMAGE_IO_HPP

