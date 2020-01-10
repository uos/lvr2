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

    PinholeCameraModel() = default;

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

    void setDistortion(const Distortion<T>& d)
    {
        m_distortion = d;
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

    Distortion<T> distortion() const
    {
       return m_distortion;
    }


private:

    /// Distortion parameters
    Distortion<T>                   m_distortion;

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