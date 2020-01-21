
#ifndef LVR2_IO_YAML_PINHOLECAMERAMODEL_IO_HPP
#define LVR2_IO_YAML_PINHOLECAMERAMODEL_IO_HPP

#include <yaml-cpp/yaml.h>
#include "lvr2/types/ScanTypes.hpp"
#include "MatrixIO.hpp"
#include "lvr2/registration/PinholeCameraModel.hpp"

namespace YAML {

/**
 * YAML-CPPs convert specialization
 * 
 * example: 
 */

// WRITE SCAN PARTIALLY
template <typename T>
struct convert<lvr2::PinholeCameraModel<T> > 
{

    /**
     * Encode Eigen matrix to yaml. 
     */
    static Node encode(const lvr2::PinholeCameraModel<T>& model) {
        Node node;

        node["type"] = "camera";
        node["camera_model"] = "pinhole";
        node["extrinsics"] = model.extrinsics();
        node["extrinsics_estimate"] = model.extrinsicsEstimate();
        node["intrinsics"] = model.intrinsics();
        node["distortion"] = model.distortion();

        return node;
    }

    static bool decode(const Node& node, lvr2::PinholeCameraModel<T>& model) 
    {

        std::string camera_model = node["camera_model"].as<std::string>();
        if(camera_model != "pinhole")
        {
            return false;
        }

        if(auto tmp = node["extrinsics"]) 
        {
            model.setExtrinsics(tmp.as<lvr2::Extrinsics<T> >() );
        } else {
            return false;
        }

        if(auto tmp = node["extrinsics_estimate"]) 
        {
            model.setExtrinsicsEstimate(tmp.as<lvr2::Extrinsics<T> >() );
        } else {
            return false;
        }

        if(auto tmp = node["intrinsics"]) 
        {
            model.setIntrinsics(tmp.as<lvr2::Intrinsics<T> >() );
        } else {
            return false;
        }

        if(auto tmp = node["distortion"]) 
        {
            model.setDistortion(tmp.as<lvr2::Distortion<T> >() );
        } else {
            return false;
        }

        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_PINHOLECAMERAMODEL_IO_HPP

