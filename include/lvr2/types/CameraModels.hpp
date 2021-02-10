#ifndef LVR2_TYPES_CAMERA_MODELS_HPP
#define LVR2_TYPES_CAMERA_MODELS_HPP

#include <vector>
#include <string>
#include <memory>
#include "MatrixTypes.hpp"

namespace lvr2 {

/**
 * @brief Interface for all CameraModels e.g. projection Models
 * 
 * Could be used to interface functions like 
 * - projectPoints
 * - getRayToPixel
 * - etc
 * 
 */
struct CameraModel {

};

struct PinholeModel : CameraModel
{
    double fx = 0;
    double fy = 0;
    double cx = 0;
    double cy = 0;
    unsigned width = 0;
    unsigned height = 0;
    std::vector<double> k;
    std::string distortionModel = "unknown";
};

using PinholeModelPtr = std::shared_ptr<PinholeModel>;

struct CylindricalModel : CameraModel {
    /// Focal length
    double                          focalLength;

    /// Offset angle
    double                          offsetAngle;

    /// Principal x, y, z
    Vector3d                        principal;

    /// Distortion
    Vector3d                        distortion;
};

using CylindricalModelPtr = std::shared_ptr<CylindricalModel>;

struct SphericalModel : CameraModel {
    /// Focal length
    double                          focalLength;

    /// Offset angle
    double                          offsetAngle;

    /// Principal x, y, z
    Vector3d                        principal;

    /// Distortion
    Vector3d                        distortion;
};

using SphericalModelPtr = std::shared_ptr<SphericalModel>;


} // namespace lvr2

#endif // LVR2_TYPES_CAMERA_MODELS_HPP