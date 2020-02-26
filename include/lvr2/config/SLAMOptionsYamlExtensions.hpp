/**
 * Copyright (c) 2020, University Osnabrück
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

/**
 * ChunkingPipeline.hpp
 *
 * @date 24.01.2020
 * @author Marcel Wiegand
 */

#ifndef LVR2_SLAM_OPTIONS_YAML_EXTENSIONS
#define LVR2_SLAM_OPTIONS_YAML_EXTENSIONS

#include <yaml-cpp/yaml.h>

#include "lvr2/registration/SLAMOptions.hpp"

namespace YAML
{
template<>
struct convert<lvr2::SLAMOptions>
{
    static Node encode(const lvr2::SLAMOptions& options)
    {
        Node node;

        // ==================== General Options ======================================================

        node["trustPose"] = options.trustPose;
        node["metascan"] = options.metascan;
        node["createFrames"] = options.createFrames;
        node["verbose"] = options.verbose;
        node["useHDF"] = options.useHDF;

        // ==================== Reduction Options ====================================================

        node["reduction"] = options.reduction;
        node["minDistance"] = options.minDistance;
        node["maxDistance"] = options.maxDistance;

        // ==================== ICP Options ==========================================================

        node["icpIterations"] = options.icpIterations;
        node["icpMaxDistance"] = options.icpMaxDistance;
        node["maxLeafSize"] = options.maxLeafSize;
        node["epsilon"] = options.epsilon;

        // ==================== SLAM Options =========================================================

        node["doLoopClosing"] = options.doLoopClosing;
        node["doGraphSLAM"] = options.doGraphSLAM;
        node["closeLoopDistance"] = options.closeLoopDistance;
        node["closeLoopPairs"] = options.closeLoopPairs;
        node["loopSize"] = options.loopSize;
        node["slamIterations"] = options.slamIterations;
        node["slamMaxDistance"] = options.slamMaxDistance;
        node["slamEpsilon"] = options.slamEpsilon;
        node["diffPosition"] = options.diffPosition;
        node["diffAngle"] = options.diffAngle;
        node["useScanOrder"] = options.useScanOrder;
        node["rotate_angle"] = options.rotate_angle;  

        return node;
    }

    static bool decode(const Node& node, lvr2::SLAMOptions& options)
    {
        if (!node.IsMap())
        {
            return false;
        }

        // ==================== General Options ======================================================

        if (node["trustPose"])
        {
            options.trustPose = node["trustPose"].as<bool>();
        }

        if (node["metascan"])
        {
            options.metascan = node["metascan"].as<bool>();
        }

        if (node["createFrames"])
        {
            options.createFrames = node["createFrames"].as<bool>();
        }

        if (node["verbose"])
        {
            options.verbose = node["verbose"].as<bool>();
        }

        if (node["useHDF"])
        {
            options.useHDF = node["useHDF"].as<bool>();
        }

        // ==================== Reduction Options ====================================================

        if (node["reduction"])
        {
            options.reduction = node["reduction"].as<double>();
        }

        if (node["minDistance"])
        {
            options.minDistance = node["minDistance"].as<double>();
        }

        if (node["maxDistance"])
        {
            options.maxDistance = node["maxDistance"].as<double>();
        }

        // ==================== ICP Options ==========================================================

        if (node["icpIterations"])
        {
            options.icpIterations = node["icpIterations"].as<int>();
        }

        if (node["icpMaxDistance"])
        {
            options.icpMaxDistance = node["icpMaxDistance"].as<double>();
        }

        if (node["maxLeafSize"])
        {
            options.maxLeafSize = node["maxLeafSize"].as<int>();
        }

        if (node["epsilon"])
        {
            options.epsilon = node["epsilon"].as<double>();
        }

        // ==================== SLAM Options =========================================================

        if (node["doLoopClosing"])
        {
            options.doLoopClosing = node["doLoopClosing"].as<bool>();
        }

        if (node["doGraphSLAM"])
        {
            options.doGraphSLAM = node["doGraphSLAM"].as<bool>();
        }

        if (node["closeLoopDistance"])
        {
            options.closeLoopDistance = node["closeLoopDistance"].as<double>();
        }

        if (node["closeLoopPairs"])
        {
            options.closeLoopPairs = node["closeLoopPairs"].as<int>();
        }

        if (node["loopSize"])
        {
            options.loopSize = node["loopSize"].as<int>();
        }

        if (node["slamIterations"])
        {
            options.slamIterations = node["slamIterations"].as<int>();
        }

        if (node["slamMaxDistance"])
        {
            options.slamMaxDistance = node["slamMaxDistance"].as<double>();
        }

        if (node["slamEpsilon"])
        {
            options.slamEpsilon = node["slamEpsilon"].as<double>();
        }

        if (node["diffPosition"])
        {
            options.diffPosition = node["diffPosition"].as<double>();
        }

        if (node["diffAngle"])
        {
            options.diffAngle = node["diffAngle"].as<double>();
        }

        if (node["useScanOrder"])
        {
            options.useScanOrder = node["useScanOrder"].as<bool>();
        }

        if (node["rotate_angle"])
        {
            options.rotate_angle = node["rotate_angle"].as<double>();
        }

        return true;
    }
};
}  // namespace YAML

#endif
