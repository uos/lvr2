
#ifndef LVR2_IO_YAML_WAVEFORM_IO_HPP
#define LVR2_IO_YAML_WAVEFORM_IO_HPP

#include <sstream>

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/scanio/yaml/Matrix.hpp"

namespace YAML {

template<>
struct convert<lvr2::Waveform> 
{

    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::Waveform& waveform) {
        
        Node node;
        node["type"] = lvr2::Waveform::type;
        node["kind"] = lvr2::Waveform::kind;
        node["maxBucketSize"] = waveform.maxBucketSize;

        return node;
    }

    static bool decode(const Node& node, lvr2::Waveform& waveform) 
    {
        if(node["type"].as<std::string>() != lvr2::Waveform::type)
        {
            return false;
        }

        if(node["type"].as<std::string>() != lvr2::Waveform::type)
        {
            return false;
        }
        
        // Get fields
        if(node["maxBucketSize"])
        {
            waveform.maxBucketSize = node["maxBucketSize"].as<int>();
        }
        else
        {
            waveform.maxBucketSize = 0; 
        }

        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_WAVEFORM_IO_HPP

