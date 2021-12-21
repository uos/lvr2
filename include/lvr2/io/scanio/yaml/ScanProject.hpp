
#ifndef LVR2_IO_YAML_SCANPROJECT_HPP
#define LVR2_IO_YAML_SCANPROJECT_HPP

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "lvr2/io/scanio/yaml/Util.hpp"

#include "AABB.hpp"
#include "Matrix.hpp"

namespace YAML {  

/**
 * YAML-CPPs convert specialization
 * 
 * example: 
 */

// WRITE SCAN PARTIALLY
template <>
struct convert<lvr2::ScanProject> 
{

    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::ScanProject& scanProj) {
        Node node;
        
        node["entity"] = lvr2::ScanProject::entity;
        node["type"] = lvr2::ScanProject::type;
        node["crs"] =  scanProj.crs;
        node["coordinate_system"] = scanProj.coordinateSystem;
        node["unit"] = scanProj.unit;
        node["transformation"] = scanProj.transformation;
        node["name"] = scanProj.name;

        if(scanProj.boundingBox)
        {
            node["aabb"] = *scanProj.boundingBox;
        }

        return node;
    }

    static bool decode(const Node& node, lvr2::ScanProject& scanProj) 
    {
        // Check if 'entity' and 'type' Tags are valid
        if (!YAML_UTIL::ValidateEntityAndType(node, 
            "scan_project", 
            lvr2::ScanProject::entity, 
            lvr2::ScanProject::type))
        {
            return false;
        }

        if(node["transformation"])
        {
            scanProj.transformation = node["transformation"].as<lvr2::Transformd>();
        }  else {
            scanProj.transformation  = lvr2::Transformd::Identity();
        }
      
        if(node["crs"])
        {
            scanProj.crs = node["crs"].as<std::string>();
        }

        if(node["coordinate_system"])
        {
            scanProj.coordinateSystem = node["coordinate_system"].as<std::string>();
        }

        if(node["unit"])
        {
            scanProj.unit = node["unit"].as<std::string>();
        }
              
        if(node["name"])
        {
            scanProj.name = node["name"].as<std::string>();
        }

        if(node["aabb"])
        {
            scanProj.boundingBox = node["aabb"].as<lvr2::BoundingBox<lvr2::BaseVector<float> > >();
        }

    
        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_SCANPROJECT_HPP

