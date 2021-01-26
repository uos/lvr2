
#ifndef LVR2_IO_YAML_SCANPROJECT_HPP
#define LVR2_IO_YAML_SCANPROJECT_HPP

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/Timestamp.hpp"

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
        
        node["type"] = lvr2::ScanProject::type;
        node["kind"] = boost::typeindex::type_id<lvr2::ScanProject>().pretty_name();
        node["crs"] =  scanProj.crs;
        node["coordinate_system"] = scanProj.coordinateSystem;
        node["unit"] = scanProj.unit;
        node["transformation"] = scanProj.transformation;
        node["name"] = scanProj.name;

        return node;
    }

    static bool decode(const Node& node, lvr2::ScanProject& scanProj) 
    {
        if(!node["type"])
        {
            std::cout << "[YAML::convert<ScanProject> - decode] 'type' Tag not found." << std::endl;
            return false;
        }

        if(node["type"].as<std::string>() != lvr2::ScanProject::type)
        {
            std::cout << "[YAML::convert<ScanProject> - decode] Try to load " << node["type"].as<std::string>() << " as " << lvr2::ScanProject::type << std::endl;
            return false;
        }

        if(node["transformation"])
        {
            scanProj.transformation = node["transformation"].as<lvr2::Transformd>();
        }  else {
            scanProj.transformation  = lvr2::Transformd::Identity();
        }
      
        if(node["coordinate_system"])
        {
            scanProj.coordinateSystem = node["coordinate_system"].as<std::string>();
        }

        if(node["unit"])
        {
            scanProj.unit = node["unit"].as<std::string>();
        } else {
            scanProj.unit = "m";
        }
              
        if(node["name"])
        {
            scanProj.name = node["name"].as<std::string>();
        }
    
        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_SCANPROJECT_HPP

