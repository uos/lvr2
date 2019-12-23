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

    /// Get the position of the camera with respect to the internally
    /// stored extrinsic matrix. This identical with returning the 
    /// camera's position in sensor coordinates.
    template<typename VecT>
    VecT position()
    {
        Eigen::Vector<3, T> p = m_extrinsics.template block<3, 1>(0, 3);
        return VecT(p[0], p[1], p[2]);
    }

    Eigen::Matrix<3, T> rotation()
    {
        return m_extrinsics.template block<3, 3>(0, 0)
    }

    Eigen::Vector<3, T> position()
    {
        return 
    }

    /// Get the position of the camera in world coordinates for a 
    /// given reference frame which is given by \ref transform. The 
    /// internally stored extrinsic information is added to \ref transform. 
    VecT position(const Transform<T>& transform)
    {
        /// TODO ...
    }

    /// Get the global rotation of the camera with respect to the 
    /// given reference frame defined by \ref transform. 
    Eigen::Matrix<3, T> rotation(const Transform<T>& transform)
    {

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
};

using CameraModelPtr = std::shared_ptr<CameraModel>;

} // namespace lvr2

#endif