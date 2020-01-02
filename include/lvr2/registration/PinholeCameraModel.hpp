#ifndef __CAMERAMODEL_HPP__
#define __CAMERAMODEL_HPP__

#include <memory>

#include <Eigen/Dense>

#include "lvr2/types/MatrixTypes.hpp"

namespace lvr2
{

template<typename T>
class PinholeCameraModel
{
public:

    PinholeCameraModel() 
    {
        for(int i = 0; i < 4; i++)
        {
            m_intrinsic_params[i] = 0.0;
        }

        for(int i = 0; i < 6; i++)
        {
            m_distortion_params[i] = 0.0;
        }
    }

    Eigen::Matrix<T, 3, 3> rotation() const
    {
        return m_extrinsics.template block<3, 3>(0, 0);
    }

    Vector3<T> position() const
    {
        return Vector3<T>(0, 0, 0); 
    }


    /// Get the global rotation of the camera with respect to the 
    /// given reference frame defined by \ref transform. 
    Eigen::Matrix<T, 3, 3> rotation(const Transform<T>& transform) const
    {
        return Eigen::Matrix<T, 3, 3>();
    }

    void setIntrinsics(const Intrinsics<T>& i)
    {
        m_intrinsics = i;
    }

    void setExtrinsics(const Extrinsics<T>& e)
    {
        m_extrinsics = e;
    }

    void setExtrinsicsEstimate(const Extrinsics<T>& e)
    {
        m_extrinsicsEstimate = e;
    }

    Intrinsics<T> intrinsics() const
    {
        return m_intrinsics;
    }

    Extrinsics<T> extrinsics() const
    {
        return m_extrinsics;
    }

    Extrinsics<T> extrinsicsEstimate() const
    {
        return m_extrinsicsEstimate;
    }

private:

    /// Intrinsic parameters in this order: fx, fy, Cx, Cy
    T                               m_intrinsic_params[4];

    /// Distortion params in this order: k1, k2, k3, k4, p1, p2
    T                               m_distortion_params[6];

    /// Intrincsic matrix
    Intrinsics<T>                   m_intrinsics;

    /// Mount calibration / extrinsic calibration
    Extrinsics<T>                   m_extrinsics;

    /// Mount calibration / extrinsic calibration
    Extrinsics<T>                   m_extrinsicsEstimate;
};

template<typename T>
using PinholeCameraModelPtr = std::shared_ptr<PinholeCameraModel<T>>;

using PinholeCameraModeld = PinholeCameraModel<double>;
using PinholeCameraModelf = PinholeCameraModel<float>;


} // namespace lvr2

#endif