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
        node["diffPoseSum"] = options.diffPoseSum;

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

        if (node["diffPoseSum"])
        {
            options.diffPoseSum = node["diffPoseSum"].as<double>();
        }

        return true;
    }
};
}  // namespace YAML

#endif
