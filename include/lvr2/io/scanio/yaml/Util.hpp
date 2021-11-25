#ifndef LVR2_IO_YAML_UTIL_HPP
#define LVR2_IO_YAML_UTIL_HPP

#include <string>
#include <yaml-cpp/yaml.h>
#include <iostream>

#include "lvr2/util/Timestamp.hpp"

namespace YAML {

inline const Node & cnode(const Node &n) {
    return n;
}
/*
Node MergeNodes(Node a, Node b)
{
    if (!b.IsMap()) {
        // If b is not a map, merge result is b, unless b is null
        return b.IsNull() ? a : b;
    }
    if (!a.IsMap()) {
        // If a is not a map, merge result is b
        return b;
    }
    if (!b.size()) {
        // If a is a map, and b is an empty map, return a
        return a;
    }
    // Create a new map 'c' with the same mappings as a, merged with b
    auto c = Node(NodeType::Map);
    for (auto n : a) {
        if (n.first.IsScalar()) {
        const std::string & key = n.first.Scalar();
        auto t = Node(cnode(b)[key]);
        if (t) {
            c[n.first] = MergeNodes(n.second, t);
            continue;
        }
        }
        c[n.first] = n.second;
    }
    // Add the mappings from 'b' not already in 'c'
    for (auto n : b) {
        if (!n.first.IsScalar() || !cnode(c)[n.first.Scalar()]) {
        c[n.first] = n.second;
        }
    }
    return c;
}
*/


} // namespace YAML

namespace YAML_UTIL
{
/**
 * @brief Check if the \ref node has a \ref tag_name Tag with value \ref required_value
 * 
 * @param node The node to check
 * @param decoder_name The decoder calling this function, used in error messages
 * @param tag_name The Tag to check for
 * @param required_value The value the Tag is supposed to have
 * @return true If the Tag exists and has the required value
 * @return false otherwise
 */
inline bool ValidateNodeTag(const YAML::Node& node, const char* decoder_name, const char* tag_name, const char* required_value)
{
    if(!node[tag_name])
    {
        std::cout << lvr2::timestamp << "[YAML::convert<" << decoder_name << "> - decode] "
                    << "Node has no '" << tag_name << "' Tag" << std::endl; 
        return false;
    }
    if (node[tag_name].as<std::string>() != required_value)
    {
        // different hierarchy level
        std::cout << lvr2::timestamp << "[YAML::convert<" << decoder_name << "> - decode] " 
                    << "Nodes " << tag_name << " '" << node[tag_name].as<std::string>()
                    << "' is not '" <<  required_value << "'" << std::endl; 
        return false;
    }

    return true;
}
/**
 * @brief Checks if the entity and type tag of the \ref node exist and match \ref entity and \ref type respectively.
 * 
 * @param node The node to check
 * @param converter_name The name of the converter calling this function. Used in the error messages.
 * @param entity The desired value the 'entity' Tag should have
 * @param type The desired value the 'type' Tag should have
 * @return true If 'entity' and 'type' exist and match the values of \ref entity and \ref type
 * @return false otherwise
 */
inline bool ValidateEntityAndType(const YAML::Node& node, const char* converter_name, const char* entity, const char* type)
{
    // If either one of these Tags is missing or does not match the Value return false
    if (!ValidateNodeTag(node, converter_name, "entity", entity) ||
        !ValidateNodeTag(node, converter_name, "type", type))
    {
        return false;
    }

    return true;
}

} // namespace YAML_UTIL

#endif // LVR2_IO_YAML_UTIL_HPP