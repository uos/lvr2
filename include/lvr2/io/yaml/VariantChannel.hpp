
#ifndef LVR2_IO_YAML_VARIANT_CHANNEL_HPP
#define LVR2_IO_YAML_VARIANT_CHANNEL_HPP

#include <sstream>

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/yaml/Matrix.hpp"
#include "lvr2/types/MultiChannelMap.hpp"

namespace YAML {

using VChannelT = lvr2::MultiChannel;
template<>
struct convert<VChannelT> 
{
    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const VChannelT& vchannel) {
        
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

    static bool decode(const Node& node, VChannelT& vchannel) 
    {
        if(node["type"].as<std::string>() != "Channel")
        {
            return false;
        }

        if(node["data_type"].as<std::string>() != vchannel.typeName())
        {
            return false;
        }
        
        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_VARIANT_CHANNEL_HPP

