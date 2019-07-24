#pragma once

#ifndef LVR2_TYPES_MULTICHANNELMAP
#define LVR2_TYPES_MULTICHANNELMAP

#include "VariantChannelMap.hpp"

namespace lvr2 {
    
//enum MultiChannelMapTypes {
//    CH_8U,
//    CH_8S,
//    CH_16U,
//    CH_16S,
//    CH_32U,
//    CH_32S,
//    CH_32F,
//    CH_64F
//};

// Don't touch the order. (ROS point_fiel compatibility)
// TODO In future these should be exchangeable.
using MultiChannelMap = VariantChannelMap<
        char,
        unsigned char,
        short,
        unsigned short,
        int,
        unsigned int,
        float,
        double
    >;

} // namespace lvr2

#endif // LVR2_TYPES_MULTICHANNELMAP
