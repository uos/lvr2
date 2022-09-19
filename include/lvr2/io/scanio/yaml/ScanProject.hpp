
#ifndef LVR2_IO_YAML_SCANPROJECT_HPP
#define LVR2_IO_YAML_SCANPROJECT_HPP

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "lvr2/util/YAMLUtil.hpp"
#include "lvr2/io/YAML.hpp"

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
//        if (!YAML_UTIL::ValidateEntityAndType(node,
//            "scan_project",
//            lvr2::ScanProject::entity,
//            lvr2::ScanProject::type))
//        {
//            return false;
//        }

        if(node["transformation"])
        {
            try {
                scanProj.transformation = node["transformation"].as<lvr2::Transformd>();
            } catch(const YAML::TypedBadConversion<lvr2::Transformd>& ex) {
                std::cerr << "[YAML - ScanProject - decode] ERROR: Could not decode 'transformation': " 
                    << node["transformation"] << " as Transformd" << std::endl; 
                return false;
            }
        }  else {
            scanProj.transformation  = lvr2::Transformd::Identity();
        }
      
        if(node["crs"])
        {
            try {
                scanProj.crs = node["crs"].as<std::string>();
            } catch(const YAML::TypedBadConversion<std::string>& ex) {
                std::cerr << "[YAML - ScanProject - decode] ERROR: Could not decode 'crs': " 
                    << node["crs"] << " as string" << std::endl; 
                return false;
            }
        }

        if(node["coordinate_system"])
        {
            try {
                scanProj.coordinateSystem = node["coordinate_system"].as<std::string>();
            } catch(const YAML::TypedBadConversion<std::string>& ex) {
                std::cerr << "[YAML - ScanProject - decode] ERROR: Could not decode 'coordinate_system': " 
                    << node["coordinate_system"] << " as string" << std::endl; 
                return false;
            }
        }

        if(node["unit"])
        {
            try {
                scanProj.unit = node["unit"].as<std::string>();
            } catch(const YAML::TypedBadConversion<std::string>& ex) {
                std::cerr << "[YAML - ScanProject - decode] ERROR: Could not decode 'unit': " 
                    << node["unit"] << " as string" << std::endl; 
                return false;
            }
        }
              
        if(node["name"])
        {
            try {
                scanProj.name = node["name"].as<std::string>();
            } catch(const YAML::TypedBadConversion<std::string>& ex) {
                std::cerr << "[YAML - ScanProject - decode] ERROR: Could not decode 'name': " 
                    << node["name"] << " as string" << std::endl; 
                return false;
            }
        }

        if(node["aabb"])
        {
            try {
                scanProj.boundingBox = node["aabb"].as<lvr2::BoundingBox<lvr2::BaseVector<float> > >();
            } catch(const YAML::TypedBadConversion<lvr2::BoundingBox<lvr2::BaseVector<float> > >& ex) {
                std::cerr << "[YAML - ScanProject - decode] ERROR: Could not decode 'aabb': " 
                    << node["aabb"] << " as BoundingBox" << std::endl; 
                return false;
            }
        }

    
        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_SCANPROJECT_HPP

