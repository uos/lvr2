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
    double fx = 0;
    double fy = 0;
    double cx = 0;
    double cy = 0;
    unsigned width = 0;
    unsigned height = 0;
    std::vector<double> k;
    std::string distortionModel = "unknown";

    // virtual Eigen::Vector2d projectPoint(const Eigen::Vector3d& P) const override 
    // {
    //     /**
    //      * [fx 0 cx]   [Px]
    //      * [0 fy cy] * [Py]
    //      * [0 0  1]    [Pz]
    //      * 
    //      * Px * fx + Pz * cx
    //      * Py * fy + Pz * cy
    //      * Pz
    //      * 
    //      * Px/Pz * fx + cx
    //      * Py/Pz * fy + cy
    //      */

    //     Eigen::Vector2d pixel;
    //     pixel(0) = fx * P(0)/P(2) + cx;
    //     pixel(1) = fy * P(1)/P(2) + cy;
    //     return pixel;
    // }
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

    // Eigen::Vector2d projectPoints(const Eigen::Vector3d& P) const override
    // {
    //     Eigen::Vector2d pixel;

    //     // x from angle: TODO


    //     // y axis like pinhole
    //     pixel(1) = focalLength * P(1)/P(2) + principal(1);

    //     return pixel;
    // }   
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