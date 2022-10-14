
#ifndef LVR2_IO_YAML_LABELMETA_IO_HPP
#define LVR2_IO_YAML_LABELMETA_IO_HPP

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "lvr2/io/YAML.hpp"
#include "lvr2/util/YAMLUtil.hpp"

using lvr2::timestamp;

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
        node["entity"] = lvr2::LabelInstance::entity;
        node["type"] = lvr2::LabelInstance::type;
        node["name"] = instance.instanceName;
        node["color"] = Load("[]");
        node["color"].push_back(instance.color[0]);
        node["color"].push_back(instance.color[1]);
        node["color"].push_back(instance.color[2]);
        return node;
    }

    static bool decode(const Node& node, lvr2::LabelInstance& instance) {

        // Check if 'entity' and 'type' Tags are valid
        if (!YAML_UTIL::ValidateEntityAndType(node, 
            "label_instance", 
            lvr2::LabelInstance::entity, 
            lvr2::LabelInstance::type))
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
        node["entity"] = lvr2::LabelInstance::entity;
        node["type"] = lvr2::LabelInstance::type;
        node["name"] = labelClass.className;

        return node;
    }

    static bool decode(const Node& node, lvr2::LabelClass& labelClass) {
        
        // Check if 'entity' and 'type' Tags are valid
        if (!YAML_UTIL::ValidateEntityAndType(node, 
            "label_class", 
            lvr2::LabelClass::entity, 
            lvr2::LabelClass::type))
        {
            return false;
        }
        labelClass.className = node["name"].as<string>();
        return true;
    }
};

}  // namespace YAML

#endif // LVR2_IO_YAML_LABELMETA_IO_HPP