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
struct CameraModel 
{
    static constexpr char           entity[] = "Model";
    // /**
    //  * @brief Project point from camera coordinate system (3D) onto image coordinate system (2D)
    //  * 
    //  * @param p 
    //  * @return Eigen::Vector2d 
    //  */
    // virtual Eigen::Vector2d projectPoint(const Eigen::Vector3d& P) const 
    // {
    //     Eigen::Vector2d pixel;
    //     return pixel;
    // }
};

struct PinholeModel : CameraModel
{
    static constexpr char           type[] = "Pinhole";

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
using PinholeModelOptional = boost::optional<PinholeModel>;

struct CylindricalModel : CameraModel 
{
    static constexpr char           type[] = "Panorama";

    /// Principal x, y
    Vector2d                        principal;

    /// Focal Length fx, fy
    Vector2d                        focalLength;

    /// FoV
    Vector2d                        fov;

    /// Distortion
    std::vector<double>             distortion;
};

using CylindricalModelPtr = std::shared_ptr<CylindricalModel>;
using CylindricalModelOptional = boost::optional<CylindricalModel>;

struct SphericalModel : CameraModel {
    static constexpr char           type[] = "Spherical";

    /// Focal length
    double                          focalLength;

    /// Offset angle
    double                          offsetAngle;

    /// Principal x, y, z
    Vector3d                        principal;

    /// Distortion
    std::vector<double>             distortion;
};

using SphericalModelPtr = std::shared_ptr<SphericalModel>;


} // namespace lvr2

#endif // LVR2_TYPES_CAMERA_MODELS_HPP