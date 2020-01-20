
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

        node["extrinsics"] = model.extrinsics();
        node["extrinsic_estimate"] = model.extrinsicsEstimate();
        node["intrinsics"] = model.intrinsics();
        node["distortion"] = model.distortion();

        return node;
    }

    static bool decode(const Node& node, lvr2::PinholeCameraModel<T>& model) {

        if(lvr2::Extrinsics<T> tmp = node["extrinsics"].as<lvr2::Extrinsics<T> >() ){
            model.setExtrinsics(tmp);
        } else {
            return false;
        }

        if(lvr2::Extrinsics<T> tmp = node["extrinsic_estimate"].as<lvr2::Extrinsics<T> >() ){
            model.setExtrinsicsEstimate(tmp);
        } else {
            return false;
        }
        
        if(lvr2::Intrinsics<T> tmp = node["intrinsics"].as<lvr2::Intrinsics<T> >() ) {
            model.setIntrinsics(tmp);
        } else {
            return false;
        }

        if(lvr2::Distortion<T> tmp = node["distortion"].as<lvr2::Distortion<T> >() ) {
            model.setDistortion(tmp);
        } else {
            return false;
        }

        return true;
    }

};

}  // namespace YAML

#endif // LVR2_IO_YAML_PINHOLECAMERAMODEL_IO_HPP

