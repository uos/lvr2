
#ifndef LVR2_IO_YAML_LABELMETA_IO_HPP
#define LVR2_IO_YAML_LABELMETA_IO_HPP

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


template <>
struct convert<lvr2::LabelInstance> 
{
    static Node encode(const lvr2::LabelInstance& instance) {
        Node node;
        node["sensor_type"] = lvr2::LabelInstance::sensorType;
        node["name"] = instance.instanceName;
        node["color"] = Load("[]");
        node["color"].push_back(instance.color[0]);
        node["color"].push_back(instance.color[1]);
        node["color"].push_back(instance.color[2]);
        return node;
    }

    static bool decode(const Node& node, lvr2::LabelInstance& instance) {
        
        if(node["sensor_type"].as<std::string>() != lvr2::LabelInstance::sensorType) 
        {
            return false;
        }

        instance.instanceName = node["name"].as<string>();
        instance.color[0] = node["color"][0].as<int>();
        instance.color[1] = node["color"][0].as<int>();
        instance.color[2] = node["color"][0].as<int>();
        return true;
    }

};

template <>
struct convert<lvr2::LabelClass> 
{
    static Node encode(const lvr2::LabelClass& labelClass) {
        Node node;
        node["sensor_type"] = lvr2::LabelInstance::sensorType;
        node["name"] = labelClass.className;

        return node;
    }

    static bool decode(const Node& node, lvr2::LabelClass& labelClass) {
        
        if(node["sensor_type"].as<std::string>() != lvr2::LabelClass::sensorType) 
        {
            return false;
        }

        labelClass.className = node["name"].as<string>();
        return true;
    }
};

}  // namespace YAML

#endif // LVR2_IO_YAML_LABELMETA_IO_HPP