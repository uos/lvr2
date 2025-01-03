#ifndef CAMERAMODELS
#define CAMERAMODELS

#include <vector>
#include <string>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <boost/optional.hpp>
#include "MatrixTypes.hpp"
#include "lvr2/util/Panic.hpp"
#include "lvr2/util/Logging.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "DistortionModels.hpp"

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
        DistortionModel distortionModel;
        // std::vector<double> distortionCoefficients;
        // std::string distortionModel = "opencv";

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

        template <typename Scalar>
        Vector2<Scalar> distortPoint(const Vector2<Scalar>& p) const
        {
            const double x = (p.x() - cx)/fx;
            const double y = (p.y() - cy)/fy;

            auto uv = distortionModel.distortPoint(Vector2d(x, y));

            return Vector2<Scalar>(uv.x() * fx + cx, uv.y() * fy + cy);
        }
    };

    using PinholeModelPtr = std::shared_ptr<PinholeModel>;
    using PinholeModelOptional = boost::optional<PinholeModel>;



    inline std::ostream& operator<<(std::ostream& os, const PinholeModel& m)
    {
        os << timestamp << "Pinhole Model" << std::endl;
        os << timestamp << "-------------" << std::endl;
        os << timestamp << "Fx: " << m.fx << std::endl;
        os << timestamp << "Fy: " << m.fy << std::endl;
        os << timestamp << "Cx: " << m.cx << std::endl;
        os << timestamp << "Cy: " << m.cy << std::endl;
        os << timestamp << "Width: " << m.width << std::endl;
        os << timestamp << "Height: " << m.height << std::endl;
        os << timestamp << "Distortion Model: " << m.distortionModel.name() << std::endl;
        for(size_t i = 0; i < m.distortionModel.coefficients().size(); i++)
        {
            os << timestamp << "Coeff  " << i << ": " << m.distortionModel.coefficients()[i] << std::endl;
        }
        
        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const PinholeModel& m)
    {

        log << lvr2::info << "[Pinhole Model] Fx: " << m.fx << lvr2::endl;
        log << "[Pinhole Model] Fy: " << m.fy << lvr2::endl;
        log << "[Pinhole Model] Cx: " << m.cx << lvr2::endl;
        log << "[Pinhole Model] Cy: " << m.cy << lvr2::endl;
        log << "[Pinhole Model] Width: " << m.width << lvr2::endl;
        log << "[Pinhole Model] Height: " << m.height << lvr2::endl;
        log << "[Pinhole Model] Distortion Model: " << m.distortionModel.name() << lvr2::endl;

        for(size_t i = 0; i < m.distortionModel.coefficients().size(); i++)
        {
            log << "[Pinhole Model] Coeff  " << i << ": " << m.distortionModel.coefficients()[i] << lvr2::endl;
        }
        
        return log;
    }


    inline std::ostream& operator<<(std::ostream& os, const PinholeModelPtr p)
    {
        if (p)
        {
            os << *p;
            os << timestamp << "[Pinhole Model] Pointer Address" << p.get() << std::endl;
        }
        else
        {
            os << timestamp << "[Pinhole Model] Nullptr" << std::endl;
        }

        return os;
    }


    inline lvr2::Logger& operator<<(lvr2::Logger& log, const PinholeModelPtr p)
    {
        if (p)
        {
            log << *p;
            log << "[Pinhole Model] Pointer Address" << p.get() << lvr2::endl;
        }
        else
        {
            log << lvr2::warning << "[Pinhole Model] Nullptr" << lvr2::endl;
        }

        return log;
    }


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

    inline std::ostream& operator<<(std::ostream& os, const CylindricalModel& m)
    {
        for(size_t i = 0; i < m.principal.size(); i++)
        {
            os << timestamp << "[Cylindrical Model] Principal " << i << ": " << m.principal[i] << std::endl;
        }
        
        for(size_t i = 0; i < m.focalLength.size(); i++)
        {
            os << timestamp << "[Cylindrical Model] FocalLength " << i << ": " << m.focalLength[i] << std::endl;
        }

        for(size_t i = 0; i < m.fov.size(); i++)
        {
            os << timestamp << "[Cylindrical Model] FOV " << i << ": " << m.fov[i] << std::endl;
        }

        os << timestamp << "[Cylindrical Model] Distortion Model: " << m.distortionModel << std::endl;

        for(size_t i = 0; i < m.distortionCoefficients.size(); i++)
        {
            os << timestamp << "[Cylindrical Model] Distortion Coefficients " << i << ": " << m.distortionCoefficients[i] << std::endl;
        }

        for(size_t i = 0; i < m.distortion.size(); i++)
        {
            os << timestamp << "[Cylindrical Model] Distortion " << i << ": " << m.distortion[i] << std::endl;
        }

        return os;
    }
    
    inline lvr2::Logger& operator<<(lvr2::Logger& log, const CylindricalModel& m)
    {
        for(size_t i = 0; i < m.principal.size(); i++)
        {
            log << "[Cylindrical Model] Principal " << i << ": " << m.principal[i] << lvr2::endl;
        }
        
        for(size_t i = 0; i < m.focalLength.size(); i++)
        {
            log << "[Cylindrical Model] FocalLength " << i << ": " << m.focalLength[i] << lvr2::endl;
        }

        for(size_t i = 0; i < m.fov.size(); i++)
        {
            log << "[Cylindrical Model] FOV " << i << ": " << m.fov[i] << lvr2::endl;
        }

        log << "[Cylindrical Model] Distortion Model: " << m.distortionModel << lvr2::endl;

        for(size_t i = 0; i < m.distortionCoefficients.size(); i++)
        {
            log << "[Cylindrical Model] Distortion Coefficients " << i << ": " << m.distortionCoefficients[i] << lvr2::endl;
        }

        for(size_t i = 0; i < m.distortion.size(); i++)
        {
            log  << "[Cylindrical Model] Distortion " << i << ": " << m.distortion[i] << lvr2::endl;
        }

        return log;
    }

    inline std::ostream& operator<<(std::ostream& os, const CylindricalModelPtr p)
    {
        if(p)
        {
            os << *p;
            os << timestamp << "[Cylindrical Model] Pointer Address" << p.get() << std::endl;
        }
        else
        {
            os << timestamp << "[Cylindrical Model] Nullptr" << std::endl;
        }

        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const CylindricalModelPtr p)
    {
        if(p)
        {
            log << *p;
            log << "[Cylindrical Model] Pointer Address" << p.get() << lvr2::endl;
        }
        else
        {
            log << lvr2::warning << "[Cylindrical Model] Nullptr" << lvr2::endl;
        }

        return log;
    }

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

    inline std::ostream& operator<<(std::ostream& os, const SphericalModel& m)
    {
        os << timestamp << "[Spherical Model] Phi: " << m.phi[0] << " " << m.phi[1] << " " << m.phi[2] << std::endl;
        os << timestamp << "[Spherical Model] Theta: " << m.theta[0] << " " << m.theta[1] << " " << m.theta[2] << std::endl;
        os << timestamp << "[Spherical Model] Range: " << m.range[0] << " " << m.range[1] << " " << m.range[2] << std::endl;
        os << timestamp << "[Spherical Model] Principal: " << m.principal << std::endl;
        os << timestamp << "[Spherical Model] Distortion Model: " << m.distortionModel << std::endl;
        for(size_t i = 0; i < m.distortionCoefficients.size(); i++)
        {
            os << timestamp << "[Spherical Model] Distortion Coefficients " << i << ": " << m.distortionCoefficients[i] << std::endl;
        }

        return os;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const SphericalModel& m)
    {
        log << lvr2::info << "[Spherical Model] Phi: " << m.phi[0] << " " << m.phi[1] << " " << m.phi[2] << lvr2::endl;
        log << "[Spherical Model] Theta: " << m.theta[0] << " " << m.theta[1] << " " << m.theta[2] << lvr2::endl;
        log << "[Spherical Model] Range: " << m.range[0] << " " << m.range[1] << " " << m.range[2] << lvr2::endl;
        log << "[Spherical Model] Principal: " << m.principal << lvr2::endl;
        log << "[Spherical Model] Distortion Model: " << m.distortionModel << lvr2::endl;
        for(size_t i = 0; i < m.distortionCoefficients.size(); i++)
        {
            log << "[Spherical Model] Distortion Coefficients " << i << ": " << m.distortionCoefficients[i] << lvr2::endl;
        }
        return log;
    }

    inline lvr2::Logger& operator<<(lvr2::Logger& log, const SphericalModelPtr p)
    {
        if(p)
        {
            log << *p;
            log << "[Spherical Model]  Pointer Address" << p.get() << lvr2::endl;
        }
        else
        {
            log << lvr2::warning << "[Spherical Model] Nullptr" << lvr2::endl;
        }

        return log;
    }


} // namespace lvr2

#endif // CAMERAMODELS
