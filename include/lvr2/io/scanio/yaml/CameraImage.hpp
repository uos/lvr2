
#ifndef LVR2_IO_YAML_CAMERAIMAGE_IO_HPP
#define LVR2_IO_YAML_CAMERAIMAGE_IO_HPP

#include <sstream>

#include <yaml-cpp/yaml.h>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "lvr2/io/scanio/yaml/Util.hpp"

#include "Matrix.hpp"

namespace YAML {

template<>
struct convert<lvr2::CameraImage> 
{
    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::CameraImage& cameraImage) { 
        Node node;

        node["entity"] = lvr2::CameraImage::entity;
        node["type"] = lvr2::CameraImage::type;
        node["transformation"] = cameraImage.transformation;
        node["extrinsics_estimation"] = cameraImage.extrinsicsEstimation;
        node["width"] = cameraImage.image.cols;
        node["height"] = cameraImage.image.rows;
        node["timestamp"] = cameraImage.timestamp;

        return node;
    }

    static bool decode(const Node& node, lvr2::CameraImage& scanImage) 
    {
       // Check if 'entity' and 'type' Tags are valid
        if (!YAML_UTIL::ValidateEntityAndType(node, 
            "CameraImage", 
            lvr2::CameraImage::entity, 
            lvr2::CameraImage::type))
        {
            return false;
        }
    
        // Get fields
        if(node["transformation"])
        {
            scanImage.transformation = node["transformation"].as<lvr2::Transformd>();
        }
        else
        {
            scanImage.transformation = lvr2::Transformd::Identity();
        }

        if(node["extrinsics_estimation"])
        {
            // NAN check?
            scanImage.extrinsicsEstimation = node["extrinsics_estimation"].as<lvr2::Extrinsicsd>();
        }
        else
        {
            scanImage.extrinsicsEstimation = lvr2::Extrinsicsd::Identity();
        }

        if(node["timestamp"])
        {
            scanImage.timestamp = node["timestamp"].as<double>();
        } 
        else 
        {
            // TODO: how to handle no timestamp?
            scanImage.timestamp = -1.0;
        }

        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_CAMERAIMAGE_IO_HPP

