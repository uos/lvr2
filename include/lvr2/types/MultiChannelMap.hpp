#pragma once

#include "VariantChannelMap.hpp"

namespace lvr2 {
    
enum MultiChannelMapTypes {
    CH_8U,
    CH_8S,
    CH_16U,
    CH_16S,
    CH_32U,
    CH_32S,
    CH_32F,
    CH_64F
};

using MultiChannelMap = VariantChannelMap<
        unsigned char,
        char,
        unsigned short,
        short,
        unsigned int,
        int,
        float,
        double
    >;

} // namespace lvr2
