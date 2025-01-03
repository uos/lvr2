#pragma once

#include <yaml-cpp/yaml.h>
#include <lvr2/texture/Material.hpp>

namespace YAML {

template<>
struct convert<lvr2::RGB8Color> 
{
    static Node encode(const lvr2::RGB8Color& color) 
    {
        
        Node node;
        node["r"] = (uint64_t) color[0];
        node["g"] = (uint64_t) color[1];
        node["b"] = (uint64_t) color[2];

        return node;
    }

    static bool decode(const Node& node, lvr2::RGB8Color& color) 
    {
        if (!node["r"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<Texture> - decode] " 
                            << "Node has no tag 'r'." << std::endl;
            return false;
        }
        if (!node["g"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<Texture> - decode] " 
                            << "Node has no tag 'g'." << std::endl;
            return false;
        }
        if (!node["b"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<Texture> - decode] " 
                            << "Node has no tag 'b'." << std::endl;
            return false;
        }

        color[0] = node["r"].as<uint64_t>();
        color[1] = node["g"].as<uint64_t>();
        color[2] = node["b"].as<uint64_t>();

        return true;
    }
};

template<>
struct convert<lvr2::Material> 
{

    static Node encode(const lvr2::Material& material) 
    {
        
        Node node;
        if (material.m_color)
        {
            node["color"] = *material.m_color;
        }
        else
        {
            node["color"] = lvr2::RGB8Color({255, 255, 255});
        }
        

        return node;
    }

    static bool decode(const Node& node, lvr2::Material& material) 
    {
        if (!node["color"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<Texture> - decode] " 
                            << "Node has no tag 'color'." << std::endl;
            return false;
        }

        lvr2::RGB8Color color = node["color"].as<lvr2::RGB8Color>();
        material.m_color = color;

        return true;
    }
};

}  // namespace YAML
