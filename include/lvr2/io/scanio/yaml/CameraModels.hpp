#ifndef LVR2_IO_YAML_CAMERA_MODELS_HPP
#define LVR2_IO_YAML_CAMERA_MODELS_HPP

#include <yaml-cpp/yaml.h>

#include "Matrix.hpp"
#include "lvr2/types/CameraModels.hpp"

namespace YAML {

template <>
struct convert<lvr2::PinholeModel>
{
    static Node encode(const lvr2::PinholeModel& model)
    {
        Node node;
        node["type"] = lvr2::PinholeModel::type;
        node["camera_model"] = lvr2::PinholeModel::kind;

        node["c"] = Load("[]");
        node["c"].push_back(model.cx);
        node["c"].push_back(model.cy);
        
        node["f"] = Load("[]");
        node["f"].push_back(model.fx);
        node["f"].push_back(model.fy);

        node["resolution"] = Load("[]");
        node["resolution"].push_back(model.width);
        node["resolution"].push_back(model.height);

        node["distortion_coefficients"] = Load("[]");
        for(size_t i = 0; i < model.k.size(); i++)
        {
            node["distortion_coefficients"].push_back(model.k[i]);
        }
        node["distortion_model"] = model.distortionModel;

        return node;
    }

    static bool decode(const Node& node, lvr2::PinholeModel& model)
    {
        if(!node["type"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<PinholeModel> - decode] "
                     << "CameraModel has no key 'type'" << std::endl; 
            return false;
        }

        if (node["type"].as<std::string>() != lvr2::PinholeModel::type)
        {
            // different hierarchy level
            std::cout << lvr2::timestamp << "[YAML::convert<PinholeModel> - decode] " 
                        << "Nodes type '" << node["type"].as<std::string>()
                        << "' is not '" <<  lvr2::PinholeModel::type << "'" << std::endl; 
            return false;
        }

        if(!node["camera_model"])
        {
            std::cout << lvr2::timestamp << "[YAML::convert<PinholeModel> - decode] "
                     << "WARNING: CameraModel has no key 'camera_model'. Assuming this model to by of kind "  << lvr2::PinholeModel::kind << std::endl;
        } else {
            if(node["camera_model"].as<std::string>() != lvr2::PinholeModel::kind)
            {
                std::cout << lvr2::timestamp << "[YAML::convert<PinholeModel> - decode] " 
                            << "Nodes camera_model '" << node["camera_model"].as<std::string>()
                            << "' is not '" <<  lvr2::PinholeModel::kind << "'" << std::endl; 
                return false;
            }
        }

        model.cx = node["c"][0].as<double>();
        model.cy = node["c"][1].as<double>();
        model.fx = node["f"][0].as<double>();
        model.fy = node["f"][1].as<double>();

        model.width = node["resolution"][0].as<unsigned int>();
        model.height = node["resolution"][1].as<unsigned int>();

        if(node["distortion_model"])
        {
            model.distortionModel = node["distortion_model"].as<std::string>();
        }

        if(node["distortion_coefficients"])
        {
            Node distortionNode = node["distortion_coefficients"];
            model.k.clear();
            if(distortionNode)
            {
                YAML::const_iterator it = distortionNode.begin();
                YAML::const_iterator it_end = distortionNode.end();

                for(; it != it_end; it++)
                {
                    model.k.push_back(it->as<double>());
                }
            }
        }

        return true;
    }
};

template <>
struct convert<lvr2::CylindricalModel>
{
    static Node encode(const lvr2::CylindricalModel& model)
    {
        Node node;

        node["type"] = lvr2::CylindricalModel::type;
        node["kind"] = lvr2::CylindricalModel::kind;

        node["principal"] = Load("[]");
        node["principal"].push_back(model.principal(0));
        node["principal"].push_back(model.principal(1));

        node["focal_length"] = Load("[]");
        node["focal_length"] = model.focalLength(0);
        node["focal_length"] = model.focalLength(1);

        node["distortion"] = Load("[]");
        for(size_t i=0; i<model.distortion.size(); i++)
        {
            node["distortion"].push_back(model.distortion[i]);
        }

        return node;
    }

    static bool decode(const Node& node, lvr2::CylindricalModel& camera)
    {
        if (node["type"].as<std::string>() != lvr2::CylindricalModel::type)
        {
            // different hierarchy level
            return false;
        }

        if(node["kind"].as<std::string>() != lvr2::CylindricalModel::kind)
        {
            // different sensor type
            return false;
        }

        return true;
    }
};

} // namespace YAML

#endif // LVR2_IO_YAML_CAMERA_MODELS_HPP