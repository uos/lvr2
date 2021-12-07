#ifndef LVR2_IO_YAML_CAMERA_MODELS_HPP
#define LVR2_IO_YAML_CAMERA_MODELS_HPP

#include <yaml-cpp/yaml.h>

#include "Matrix.hpp"
#include "lvr2/types/CameraModels.hpp"
#include "lvr2/io/scanio/yaml/Util.hpp"

namespace YAML {

template <>
struct convert<lvr2::PinholeModel>
{
    static Node encode(const lvr2::PinholeModel& model)
    {
        Node node;
        node["entity"] = lvr2::PinholeModel::entity;
        node["type"] = lvr2::PinholeModel::type;

        

        // Old
        // node["c"] = Load("[]");
        // node["c"].push_back(model.cx);
        // node["c"].push_back(model.cy);
        
        // node["f"] = Load("[]");
        // node["f"].push_back(model.fx);
        // node["f"].push_back(model.fy);

        // New
        Eigen::Matrix3d M;
        M.setIdentity();

        M(0,0) = model.fx;
        M(1,1) = model.fy;

        M(0,2) = model.cx;
        M(1,2) = model.cy;

        node["intrinsics"] = M;

        // Optional?
        node["resolution"] = Load("[]");
        node["resolution"].push_back(model.width);
        node["resolution"].push_back(model.height);

        // Distortion
        node["distortion_model"] = model.distortionModel;

        node["distortion_coefficients"] = Load("[]");
        for(size_t i = 0; i < model.distortionCoefficients.size(); i++)
        {
            node["distortion_coefficients"].push_back(model.distortionCoefficients[i]);
        }
        

        return node;
    }

    static bool decode(const Node& node, lvr2::PinholeModel& model)
    {
        // TODO: Deprecation check for camera_model Tag
        // Check if 'entity' and 'type' Tags are valid
        if (!YAML_UTIL::ValidateEntityAndType(node, 
            "pinhole", 
            lvr2::PinholeModel::entity, 
            lvr2::PinholeModel::type))
        {
            return false;
        }

        // Old
        // model.cx = node["c"][0].as<double>();
        // model.cy = node["c"][1].as<double>();
        // model.fx = node["f"][0].as<double>();
        // model.fy = node["f"][1].as<double>();

        // New
        if(node["intrinsics"])
        {
            Eigen::Matrix3d M = node["intrinsics"].as<Eigen::Matrix3d>();
            model.fx = M(0,0);
            model.fy = M(1,1);
            model.cx = M(0,2);
            model.cy = M(1,2);
        }
        
        if(node["resolution"])
        {
            model.width = node["resolution"][0].as<unsigned int>();
            model.height = node["resolution"][1].as<unsigned int>();
        }
        

        if(node["distortion_model"])
        {
            model.distortionModel = node["distortion_model"].as<std::string>();
        }

        if(node["distortion_coefficients"])
        {
            Node distortionNode = node["distortion_coefficients"];
            model.distortionCoefficients.clear();
            if(distortionNode)
            {
                YAML::const_iterator it = distortionNode.begin();
                YAML::const_iterator it_end = distortionNode.end();

                for(; it != it_end; it++)
                {
                    model.distortionCoefficients.push_back(it->as<double>());
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

        node["entity"] = lvr2::CylindricalModel::entity;
        node["type"] = lvr2::CylindricalModel::type;

        node["principal_point"] = Load("[]");
        node["principal_point"].push_back(model.principal(0));
        node["principal_point"].push_back(model.principal(1));

        node["focal_lengths"] = Load("[]");
        node["focal_lengths"].push_back(model.focalLength(0));
        node["focal_lengths"].push_back(model.focalLength(1));

        node["camera_fov"] = Load("[]");
        node["camera_fov"].push_back(model.fov(0));
        node["camera_fov"].push_back(model.fov(1));

        node["distortion_model"] = model.distortionModel;
        node["distortion_coefficients"] = Load("[]");
        for(size_t i=0; i<model.distortionCoefficients.size(); i++)
        {
            node["distortion_coefficients"].push_back(model.distortionCoefficients[i]);
        }

        return node;
    }

    static bool decode(const Node& node, lvr2::CylindricalModel& model)
    {
        // Check if 'entity' and 'type' Tags are valid
        if (!YAML_UTIL::ValidateEntityAndType(node, 
            "cylindrical", 
            lvr2::CylindricalModel::entity, 
            lvr2::CylindricalModel::type))
        {
            return false;
        }

        model.principal(0) = node["principal_point"][0].as<double>();
        model.principal(1) = node["principal_point"][1].as<double>();

        model.focalLength(0) = node["focal_lengths"][0].as<double>();
        model.focalLength(1) = node["focal_lengths"][1].as<double>();

        model.fov(0) = node["camera_fov"][0].as<double>();
        model.fov(1) = node["camera_fov"][1].as<double>();

        
        
        model.distortionModel = node["distortion_model"].as<std::string>();

        model.distortionCoefficients.resize(0);
        Node distortionNode = node["distortion_coefficients"];

        if(distortionNode)
        {
            YAML::const_iterator it = distortionNode.begin();
            YAML::const_iterator it_end = distortionNode.end();

            for(; it != it_end; it++)
            {
                model.distortionCoefficients.push_back(it->as<double>());
            }
        }


        return true;
    }
};

template <>
struct convert<lvr2::SphericalModel>
{
    static Node encode(const lvr2::SphericalModel& model)
    {
        Node node;

        node["entity"] = lvr2::SphericalModel::entity;
        node["type"] = lvr2::SphericalModel::type;

        node["phi"] = Load("[]");
        node["phi"].push_back(model.phi[0]);
        node["phi"].push_back(model.phi[1]);
        node["phi"].push_back(model.phi[2]);

        node["theta"] = Load("[]");
        node["theta"].push_back(model.theta[0]);
        node["theta"].push_back(model.theta[1]);
        node["theta"].push_back(model.theta[2]);

        node["range"] = Load("[]");
        node["range"].push_back(model.range[0]);
        node["range"].push_back(model.range[1]);
        node["range"].push_back(model.range[2]);

        node["principal_point"] = Load("[]");
        node["principal_point"].push_back(model.principal(0));
        node["principal_point"].push_back(model.principal(1));
        node["principal_point"].push_back(model.principal(2));


        node["distortion_model"] = model.distortionModel;
        node["distortion_coefficients"] = Load("[]");
        for(size_t i=0; i<model.distortionCoefficients.size(); i++)
        {
            node["distortion_coefficients"].push_back(model.distortionCoefficients[i]);
        }

        return node;
    }

    static bool decode(const Node& node, lvr2::SphericalModel& model)
    {
        // Check if 'entity' and 'type' Tags are valid
        if (!YAML_UTIL::ValidateEntityAndType(node, 
            "spherical", 
            lvr2::SphericalModel::entity, 
            lvr2::SphericalModel::type))
        {
            return false;
        }

        
        if(node["phi"])
        {
            model.phi[0] = node["phi"][0].as<double>();
            model.phi[1] = node["phi"][1].as<double>();
            model.phi[2] = node["phi"][2].as<double>();
        }

        if(node["theta"])
        {
            model.theta[0] = node["theta"][0].as<double>();
            model.theta[1] = node["theta"][1].as<double>();
            model.theta[2] = node["theta"][2].as<double>();
        }

        if(node["range"])
        {
            model.range[0] = node["range"][0].as<double>();
            model.range[1] = node["range"][1].as<double>();
            model.range[2] = node["range"][2].as<double>();
        }

        if(node["principal_point"])
        {
            model.principal(0) = node["principal_point"][0].as<double>();
            model.principal(1) = node["principal_point"][1].as<double>();
            model.principal(2) = node["principal_point"][2].as<double>();
        }
        
        if(node["distortion_model"])
        {
            model.distortionModel = node["distortion_model"].as<std::string>();
        }

        
        Node distortionNode = node["distortion_coefficients"];
        model.distortionCoefficients.resize(0);

        if(distortionNode)
        {
            YAML::const_iterator it = distortionNode.begin();
            YAML::const_iterator it_end = distortionNode.end();

            for(; it != it_end; it++)
            {
                model.distortionCoefficients.push_back(it->as<double>());
            }
        }


        return true;
    }
};


} // namespace YAML

#endif // LVR2_IO_YAML_CAMERA_MODELS_HPP