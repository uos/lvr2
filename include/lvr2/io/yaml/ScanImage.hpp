
#ifndef LVR2_IO_YAML_SCANIMAGE_IO_HPP
#define LVR2_IO_YAML_SCANIMAGE_IO_HPP

#include <sstream>

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/yaml/MatrixIO.hpp"

namespace YAML {

template<>
struct convert<lvr2::ScanImage> 
{

    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::ScanImage& scanImage) {
        
        Node node;
        node["sensor_type"] = lvr2::ScanImage::sensorType;
        node["extrinsics"] = scanImage.extrinsics;
        node["extrinsics_estimate"] = scanImage.extrinsicsEstimate;
        node["width"] = scanImage.image.cols;
        node["height"] = scanImage.image.rows;
        node["image_file"] = scanImage.imageFile.string();

        return node;
    }

    static bool decode(const Node& node, lvr2::ScanImage& scanImage) 
    {
        if(node["sensor_type"].as<std::string>() != lvr2::ScanImage::sensorType)
        {
            return false;
        }
        
        // Get fields
        if(node["extrinsics"])
        {
            scanImage.extrinsics = node["extrinsics"].as<lvr2::Extrinsicsd>();
        }
        else
        {
            scanImage.extrinsics = lvr2::Extrinsicsd::Identity();
        }

        if(node["extrinsics_estimate"])
        {
            scanImage.extrinsicsEstimate = node["extrinsics"].as<lvr2::Extrinsicsd>();
        }
        else
        {
            scanImage.extrinsicsEstimate = lvr2::Extrinsicsd::Identity();
        }

        // Makes no sense to read with and height here...
       
        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_PINHOLECAMERAMODEL_IO_HPP

