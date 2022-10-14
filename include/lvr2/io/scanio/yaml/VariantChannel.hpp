
#ifndef LVR2_IO_YAML_VARIANT_CHANNEL_HPP
#define LVR2_IO_YAML_VARIANT_CHANNEL_HPP

#include <yaml-cpp/yaml.h>
#include <utility>

#include "lvr2/io/YAML.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "lvr2/types/MultiChannelMap.hpp"
#include "lvr2/util/YAMLUtil.hpp"

using lvr2::timestamp;

namespace YAML
{

    template <>
    struct convert<lvr2::MultiChannel>
    {
        /**
         * Encode Eigen matrix to yaml.
         */
        static Node encode(const lvr2::MultiChannel &vchannel)
        {

            Node node;

            std::string type = "array";

            node["entity"] = "channel";
            node["data_type"] = vchannel.typeName();

            node["type"] = type;

            if (type == "custom")
            {
                // for custom
                node["stored_type"] = lvr2::Channel<unsigned char>::typeName();
            }

            node["shape"] = Load("[]");
            node["shape"].push_back(vchannel.numElements());
            node["shape"].push_back(vchannel.width());

            return node;
        }

        static bool decode(const Node &node, lvr2::MultiChannel &vchannel)
        {
            // Check if 'entity' and 'type' Tags are valid
            // TODO: Channels dont have entity and type fields,
            // if that changes this needs to be changed too
            if (!YAML_UTIL::ValidateEntityAndType(node,
                                                  "multi_channel",
                                                  "channel",
                                                  "array"))
            {
                return false;
            }
            // if(!node["type"])
            // {
            //     std::cout << lvr2::timestamp << "[YAML::convert<MultiChannel> - decode] "
            //              << "MultiChannel meta has no 'type'" << std::endl;
            //     return false;
            // }

            // if(node["entity"] && node["entity"].as<std::string>() != "Channel")
            // {
            //     std::cout << lvr2::timestamp << "[YAML::convert<MultiChannel> - decode] "
            //                 << "Nodes type '" << node["entity"].as<std::string>()
            //                 << "' is not 'Channel'" << std::endl;
            // }

            // if(node["type"] && node["type"].as<std::string>() != "Channel")
            // {
            //     std::cout << lvr2::timestamp << "[YAML::convert<MultiChannel> - decode] "
            //                 << "Nodes type '" << node["type"].as<std::string>()
            //                 << "' is not 'Channel'" << std::endl;
            //     return false;
            // }

            // if(node["data_type"].as<std::string>() != vchannel.typeName())
            // {
            //     return false;
            // }

            return true;
        }
    };

    // template<>
    // struct convert< std::pair<std::string, lvr2::MultiChannel> >
    // {
    //     /**
    //      * Encode Eigen matrix to yaml.
    //      */
    //     static Node encode(const std::pair<std::string, lvr2::MultiChannel>& vit) {

    //         Node node;
    //         // node["name"] = vit.first;
    //         return node;
    //     }

    //     static bool decode(const Node& node, std::pair<std::string, lvr2::MultiChannel>& vit)
    //     {

    //         return true;
    //     }

    // };

} // namespace YAML

#endif // LVR2_IO_YAML_VARIANT_CHANNEL_HPP
