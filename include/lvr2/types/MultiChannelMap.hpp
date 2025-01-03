/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#ifndef LVR2_TYPES_MULTICHANNELMAP
#define LVR2_TYPES_MULTICHANNELMAP

#include "VariantChannelMap.hpp"
#include "CustomChannelTypes.hpp"

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
//};s

// Don't touch the order. (ROS point_fiel compatibility)
// TODO In future these should be exchangeable.
using MultiChannelMap = VariantChannelMap<
        char,
        unsigned char,
        short,
        unsigned short,
        int,
        unsigned int,
        long int,
        unsigned long int,
        float,
        double,
        bool,
        WaveformData
    >;

using MultiChannel = typename MultiChannelMap::val_type;
using MultiChannelOptional = boost::optional<MultiChannel>;

} // namespace lvr2

#endif // LVR2_TYPES_MULTICHANNELMAP
