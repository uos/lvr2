#ifndef LVR2_LSR_OPTIONS_YAML_EXTENSIONS
#define LVR2_LSR_OPTIONS_YAML_EXTENSIONS

#include <yaml-cpp/yaml.h>

#include "lvr2/reconstruction/LargeScaleReconstruction.hpp"

namespace YAML
{
template<>
struct convert<lvr2::LSROptions>
{
    static Node encode(const lvr2::LSROptions& options)
    {
        Node node;

        node["filePath"] = options.filePath;
        node["voxelSize"] = options.voxelSize;
        node["bgVoxelSize"] = options.bgVoxelSize;
        node["scale"] = options.scale;
        node["chunkSize"] = options.chunkSize;
        node["nodeSize"] = options.nodeSize;
        node["partMethod"] = options.partMethod;
        node["Ki"] = options.Ki;
        node["Kd"] = options.Kd;
        node["Kn"] = options.Kn;
        node["useRansac"] = options.useRansac;
        node["extrude"] = options.extrude;
        node["removeDanglingArtifacts"] = options.removeDanglingArtifacts;
        node["cleanContours"] = options.cleanContours;
        node["fillHoles"] = options.fillHoles;
        node["optimizePlanes"] = options.optimizePlanes;
        node["getNormalThreshold"] = options.getNormalThreshold;
        node["planeIterations"] = options.planeIterations;
        node["MinPlaneSize"] = options.MinPlaneSize;
        node["SmallRegionThreshold"] = options.SmallRegionThreshold;
        node["retesselate"] = options.retesselate;
        node["LineFusionThreshold"] = options.LineFusionThreshold;

        return node;
    }

    static bool decode(const Node& node, lvr2::LSROptions& options)
    {
        if (!node.IsMap())
        {
            return false;
        }

        if (node["filePath"])
        {
            options.filePath = node["filePath"].as<std::string>();
        }

        if (node["voxelSize"])
        {
            options.voxelSize = node["voxelSize"].as<float>();
        }

        if (node["bgVoxelSize"])
        {
            options.bgVoxelSize = node["bgVoxelSize"].as<float>();
        }

        if (node["scale"])
        {
            options.scale = node["scale"].as<float>();
        }

        if (node["chunkSize"])
        {
            options.chunkSize = node["chunkSize"].as<size_t>();
        }

        if (node["nodeSize"])
        {
            options.nodeSize = node["nodeSize"].as<uint>();
        }

        if (node["partMethod"])
        {
            options.partMethod = node["partMethod"].as<int>();
        }

        if (node["Ki"])
        {
            options.Ki = node["Ki"].as<int>();
        }

        if (node["Kd"])
        {
            options.Kd = node["Kd"].as<int>();
        }

        if (node["Kn"])
        {
            options.Kn = node["Kn"].as<int>();
        }

        if (node["useRansac"])
        {
            options.useRansac = node["useRansac"].as<bool>();
        }

        if (node["extrude"])
        {
            options.extrude = node["extrude"].as<bool>();
        }

        if (node["removeDanglingArtifacts"])
        {
            options.removeDanglingArtifacts = node["removeDanglingArtifacts"].as<int>();
        }

        if (node["cleanContours"])
        {
            options.cleanContours = node["cleanContours"].as<int>();
        }

        if (node["fillHoles"])
        {
            options.fillHoles = node["fillHoles"].as<int>();
        }

        if (node["optimizePlanes"])
        {
            options.optimizePlanes = node["optimizePlanes"].as<bool>();
        }

        if (node["getNormalThreshold"])
        {
            options.getNormalThreshold = node["getNormalThreshold"].as<float>();
        }

        if (node["planeIterations"])
        {
            options.planeIterations = node["planeIterations"].as<int>();
        }

        if (node["MinPlaneSize"])
        {
            options.MinPlaneSize = node["MinPlaneSize"].as<int>();
        }

        if (node["SmallRegionThreshold"])
        {
            options.SmallRegionThreshold = node["SmallRegionThreshold"].as<int>();
        }

        if (node["retesselate"])
        {
            options.retesselate = node["retesselate"].as<bool>();
        }

        if (node["LineFusionThreshold"])
        {
            options.LineFusionThreshold = node["LineFusionThreshold"].as<float>();
        }

        return true;
    }
};
}  // namespace YAML

#endif
