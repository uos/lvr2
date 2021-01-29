
#ifndef LVR2_IO_YAML_VARIANT_CHANNEL_HPP
#define LVR2_IO_YAML_VARIANT_CHANNEL_HPP

#include <yaml-cpp/yaml.h>

#include "Matrix.hpp"
#include "lvr2/io/Timestamp.hpp"

#include "lvr2/types/MultiChannelMap.hpp"


namespace YAML {

template<>
struct convert<lvr2::MultiChannel> 
{
    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::MultiChannel& vchannel) {
        
        Node node;

        std::string kind = "basic";

        node["type"] = "Channel";
        node["data_type"] = vchannel.typeName();

        node["kind"] = kind;
        
        if(kind == "custom")
        {
            // for custom 
            node["stored_type"] = lvr2::Channel<unsigned char>::typeName();
        }

        node["shape"] = Load("[]");
        node["shape"].push_back(vchannel.numElements());
        node["shape"].push_back(vchannel.width());

        return node;
    }

    static bool decode(const Node& node, lvr2::MultiChannel& vchannel) 
    {
        if(!node["type"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<MultiChannel> - decode] "
                     << "MultiChannel meta has no 'type'" << std::endl; 
            return false;
        }

        if(node["type"].as<std::string>() != "Channel")
        {
            std::cout << lvr2::timestamp << "[YAML::convert<MultiChannel> - decode] " 
                        << "Nodes type '" << node["type"].as<std::string>()
                        << "' is not 'Channel'" << std::endl; 
            return false;
        }

        // if(node["data_type"].as<std::string>() != vchannel.typeName())
        // {
        //     return false;
        // }
        
        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_VARIANT_CHANNEL_HPP

