
#ifndef LVR2_IO_YAML_VARIANT_CHANNEL_HPP
#define LVR2_IO_YAML_VARIANT_CHANNEL_HPP

#include <sstream>

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/yaml/MatrixIO.hpp"
#include "lvr2/types/BaseBuffer.hpp"

namespace YAML {

using VChannelT = lvr2::BaseBuffer::val_type;

template<>
struct convert<VChannelT> 
{
    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const VChannelT& vchannel) {
        
        Node node;

        node["sensor_type"] = "Channel";
        node["channel_type"] = vchannel.type();
        node["dims"] = Load("[]");
        node["dims"].push_back(vchannel.numElements());
        node["dims"].push_back(vchannel.width());

        return node;
    }

    static bool decode(const Node& node, VChannelT& vchannel) 
    {
        if(node["sensor_type"].as<std::string>() != "Channel")
        {
            return false;
        }

        if(node["channel_type"].as<int>() != vchannel.type())
        {
            return false;
        }
        
        // Makes no sense to read with and height here...
       
        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_VARIANT_CHANNEL_HPP

