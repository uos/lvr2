
#ifndef LVR2_IO_YAML_WAVEFORM_IO_HPP
#define LVR2_IO_YAML_WAVEFORM_IO_HPP

#include <sstream>

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/io/yaml/MatrixIO.hpp"

namespace YAML {

template<>
struct convert<lvr2::Waveform> 
{

    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::Waveform& waveform) {
        
        Node node;
        node["sensor_type"] = lvr2::Waveform::sensorType;
        node["maxBucketSize"] = waveform.maxBucketSize;

        return node;
    }

    static bool decode(const Node& node, lvr2::Waveform& waveform) 
    {
        if(node["sensor_type"].as<std::string>() != lvr2::Waveform::sensorType)
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

