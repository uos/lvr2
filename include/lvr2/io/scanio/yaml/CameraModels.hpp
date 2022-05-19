#ifndef LVR2_IO_YAML_CAMERA_MODELS_HPP
#define LVR2_IO_YAML_CAMERA_MODELS_HPP

#include <yaml-cpp/yaml.h>

#include "lvr2/io/YAML.hpp"
#include "lvr2/types/CameraModels.hpp"
#include "lvr2/util/YAMLUtil.hpp"

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

        node["intrinsic"] = M;

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
        if(node["intrinsic"])
        {
            try {
                Eigen::Matrix3d M = node["intrinsic"].as<Eigen::Matrix3d>();
                model.fx = M(0,0);
                model.fy = M(1,1);
                model.cx = M(0,2);
                model.cy = M(1,2);
            } catch(const YAML::TypedBadConversion<Eigen::Matrix3d>& ex) {
                std::cerr << "[YAML - PinholeModel - decode] ERROR: Could not decode 'intrinsic': "
                    << node["intrinsic"] << " as Eigen::Matrix3d" << std::endl; 
                return false;
            }
        }
        
        if(node["resolution"])
        {
            try {
                model.width = node["resolution"][0].as<unsigned int>();
                model.height = node["resolution"][1].as<unsigned int>();
            } catch(const YAML::TypedBadConversion<unsigned int>& ex) {
                std::cerr << "[YAML - PinholeModel - decode] ERROR: Could not decode 'resolution': "
                    << node["resolution"] << " as 2 unsigned ints" << std::endl; 
                return false;
            }
        }

        if(node["distortion_model"])
        {
            try {
                model.distortionModel = node["distortion_model"].as<std::string>();
            } catch(const YAML::TypedBadConversion<std::string>& ex) {
                std::cerr << "[YAML - PinholeModel - decode] ERROR: Could not decode 'distortion_model': "
                    << node["distortion_model"] << " as string" << std::endl; 
                return false;
            }
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
                    try {
                        model.distortionCoefficients.push_back(it->as<double>());
                    } catch(const YAML::TypedBadConversion<double>& ex) {
                        std::cerr << "[YAML - PinholeModel - decode] ERROR: Could not decode 'distortion_coefficients' entry: "
                            << *it << " as double" << std::endl;
                        return false;
                    }
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
        node["principal_point"].push_back(model.principal[0]);
        node["principal_point"].push_back(model.principal[1]);

        node["focal_lengths"] = Load("[]");
        node["focal_lengths"].push_back(model.focalLength[0]);
        node["focal_lengths"].push_back(model.focalLength[1]);

        node["camera_fov"] = Load("[]");
        node["camera_fov"].push_back(model.fov[0]);
        node["camera_fov"].push_back(model.fov[1]);

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

        if(node["principal_point"])
        {
            try {
                model.principal[0] = node["principal_point"][0].as<double>();
                model.principal[1] = node["principal_point"][1].as<double>();
            } catch(const YAML::TypedBadConversion<double>& ex) {
                std::cerr << "[YAML - CylindricalModel - decode] ERROR: Could not decode 'principal_point': "
                    << node["principal_point"] << " as 2 doubles" << std::endl; 
                return false;
            }
        } else {
            std::cout << "[YAML - CylindricalModel - decode] WARNING: Field 'principal_point' not found." << std::endl; 
        }
        
        if(node["focal_lengths"])
        {
            try {
                model.focalLength[0] = node["focal_lengths"][0].as<double>();
                model.focalLength[1] = node["focal_lengths"][1].as<double>();
            } catch(const YAML::TypedBadConversion<double>& ex) {
                std::cerr << "[YAML - CylindricalModel - decode] ERROR: Could not decode 'focal_lengths': "
                    << node["focal_lengths"] << " as 2 doubles" << std::endl; 
                return false;
            }
        } else {
            std::cout << "[YAML - CylindricalModel - decode] WARNING: Field 'focal_lengths' not found." << std::endl; 
        }
        

        if(node["camera_fov"])
        {
            try {
                model.fov[0] = node["camera_fov"][0].as<double>();
                model.fov[1] = node["camera_fov"][1].as<double>();
            } catch(const YAML::TypedBadConversion<double>& ex) {
                std::cerr << "[YAML - CylindricalModel - decode] ERROR: Could not decode 'camera_fov': "
                    << node["camera_fov"] << " as 2 doubles" << std::endl; 
                return false;
            }
        } else {
            std::cout << "[YAML - CylindricalModel - decode] WARNING: Field 'camera_fov' not found." << std::endl;
        }
        
        if(node["distortion_model"])
        {
            try {
                model.distortionModel = node["distortion_model"].as<std::string>();
            } catch(const YAML::TypedBadConversion<std::string>& ex) {
                std::cerr << "[YAML - CylindricalModel - decode] ERROR: Could not decode 'distortion_model': "
                    << node["distortion_model"] << " as string" << std::endl; 
                return false;
            }
        } else {
            std::cout << "[YAML - CylindricalModel - decode] WARNING: Field 'distortion_model' not found." << std::endl;
        }
        

        model.distortionCoefficients.resize(0);
        Node distortionNode = node["distortion_coefficients"];

        if(distortionNode)
        {
            YAML::const_iterator it = distortionNode.begin();
            YAML::const_iterator it_end = distortionNode.end();

            for(; it != it_end; it++)
            {
                try {
                    model.distortionCoefficients.push_back(it->as<double>());
                } catch(const YAML::TypedBadConversion<double>& ex) {
                    std::cerr << "[YAML - CylindricalModel - decode] ERROR: Could not decode 'distortion_coefficients' entry: "
                        << *it << " as double" << std::endl; 
                    return false;
                }
            }
        } else {
            std::cout << "[YAML - CylindricalModel - decode] WARNING: Field 'distortion_coefficients' not found." << std::endl;
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
            try {
                model.phi[0] = node["phi"][0].as<double>();
                model.phi[1] = node["phi"][1].as<double>();
                model.phi[2] = node["phi"][2].as<double>();
            } catch(const YAML::TypedBadConversion<double>& ex) {
                std::cerr << "[YAML - SphericalModel - decode] ERROR: Could not decode 'phi': "
                    << node["phi"] << " as 3 doubles" << std::endl; 
                return false;
            }
        }

        if(node["theta"])
        {
            try {
                model.theta[0] = node["theta"][0].as<double>();
                model.theta[1] = node["theta"][1].as<double>();
                model.theta[2] = node["theta"][2].as<double>();
            } catch(const YAML::TypedBadConversion<double>& ex) {
                std::cerr << "[YAML - SphericalModel - decode] ERROR: Could not decode 'theta': "
                    << node["theta"] << " as 3 doubles" << std::endl; 
                return false;
            }
        }

        if(node["range"])
        {
            try {
                model.range[0] = node["range"][0].as<double>();
                model.range[1] = node["range"][1].as<double>();
                model.range[2] = node["range"][2].as<double>();
            } catch(const YAML::TypedBadConversion<double>& ex) {
                std::cerr << "[YAML - SphericalModel - decode] ERROR: Could not decode 'range': "
                    << node["range"] << " as 3 doubles" << std::endl; 
                return false;
            }
        }

        if(node["principal_point"])
        {
            try {
                model.principal(0) = node["principal_point"][0].as<double>();
                model.principal(1) = node["principal_point"][1].as<double>();
                model.principal(2) = node["principal_point"][2].as<double>();
            } catch(const YAML::TypedBadConversion<double>& ex) {
                std::cerr << "[YAML - SphericalModel - decode] ERROR: Could not decode 'principal_point': "
                    << node["principal_point"] << " as 3 doubles" << std::endl; 
                return false;
            }
        }
        
        if(node["distortion_model"])
        {
            try {
                model.distortionModel = node["distortion_model"].as<std::string>();
            } catch(const YAML::TypedBadConversion<std::string>& ex) {
                std::cerr << "[YAML - SphericalModel - decode] ERROR: Could not decode 'distortion_model': "
                    << node["distortion_model"] << " as string" << std::endl; 
                return false;
            }
        }

        Node distortionNode = node["distortion_coefficients"];
        model.distortionCoefficients.resize(0);

        if(distortionNode)
        {
            YAML::const_iterator it = distortionNode.begin();
            YAML::const_iterator it_end = distortionNode.end();

            for(; it != it_end; it++)
            {
                try {
                    model.distortionCoefficients.push_back(it->as<double>());
                } catch(const YAML::TypedBadConversion<double>& ex) {
                    std::cerr << "[YAML - SphericalModel - decode] ERROR: Could not decode 'distortion_coefficients' entry: "
                        << *it << " as double" << std::endl; 
                    return false;
                }
            }
        }


        return true;
    }
};


} // namespace YAML

#endif // LVR2_IO_YAML_CAMERA_MODELS_HPP