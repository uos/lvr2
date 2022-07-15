#ifndef LVR2_TYPES_CAMERA_MODELS_HPP
#define LVR2_TYPES_CAMERA_MODELS_HPP

#include <vector>
#include <string>
#include <memory>
#include "MatrixTypes.hpp"
#include "lvr2/util/Panic.hpp"

namespace lvr2
{

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
        static constexpr char entity[] = "model";
        /**
         * @brief Project point from camera coordinate system (3D) onto image coordinate system (2D)
         *
         * @param p
         * @return Eigen::Vector2d
         */
        virtual Eigen::Vector2f projectPoint(const Eigen::Vector3f& P) const
        {
            panic_unimplemented("[CameraModel] projectPoint() needs to be overriden by subclass");
            Eigen::Vector2f pixel;
            return pixel;
        }
    };

    struct PinholeModel : CameraModel
    {
        static constexpr char type[] = "pinhole";

        double fx = 0;
        double fy = 0;
        double cx = 0;
        double cy = 0;
        unsigned width = 0;
        unsigned height = 0;

        /// Distortion
        std::vector<double> distortionCoefficients;
        std::string distortionModel = "opencv";

        Eigen::Vector2f projectPoint(const Eigen::Vector3f& p) const override
        {
            Eigen::Matrix3f camMat = Eigen::Matrix3f::Zero();
            camMat(0, 0) = fx;
            camMat(0, 2) = cx;
            camMat(1, 1) = fy;
            camMat(1, 2) = cy;
            camMat(2, 2) = 1;

            // double u = fx * p.x() + cx * p.z();
            // double v = fy * p.y() + cy * p.z();
            // u /= p.z();
            // v /= p.z();
            Eigen::Vector3f proj = camMat * p;
            // TODO: distort or not?
            return Eigen::Vector2f(proj.x() / proj.z(), proj.y() / proj.z());
        };
    };

    using PinholeModelPtr = std::shared_ptr<PinholeModel>;
    using PinholeModelOptional = boost::optional<PinholeModel>;

    struct CylindricalModel : CameraModel
    {
        static constexpr char type[] = "cylindrical";

        /// Principal x, y
        std::vector<double> principal;

        /// Focal Length fx, fy
        std::vector<double> focalLength;

        /// FoV
        std::vector<double> fov;

        /// Distortion
        std::vector<double> distortionCoefficients;
        std::string distortionModel = "schneider_maass";
        std::vector<double> distortion;
    };

    using CylindricalModelPtr = std::shared_ptr<CylindricalModel>;
    using CylindricalModelOptional = boost::optional<CylindricalModel>;

    struct SphericalModel : CameraModel
    {
        static constexpr char type[] = "spherical";

        /// Phi: min, max, inc
        double phi[3] = {0.0, 0.0, 0.0};

        /// Theta: min, max, inc
        double theta[3] = {0.0, 0.0, 0.0};

        /// Range: min, max, inc
        double range[3] = {0.0, 0.0, 0.0};

        /// Principal x, y, z
        Vector3d principal = Vector3d(0.0, 0.0, 0.0);

        /// Distortion
        std::vector<double> distortionCoefficients;
        std::string distortionModel = "unknown";
    };

    using SphericalModelPtr = std::shared_ptr<SphericalModel>;
    using SphericalModelOptional = boost::optional<SphericalModel>;

} // namespace lvr2

#endif // LVR2_TYPES_CAMERA_MODELS_HPP