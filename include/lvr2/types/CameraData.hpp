#ifndef CAMDATA_HPP_
#define CAMDATA_HPP_

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <vector>

namespace lvr2
{

/**
 * @brief   Struct to hold a camera image together with intrinsic 
 *          and extrinsic camera parameters
 * 
 */
struct CameraData
{
    /// Instrinsic camera paramter matrix
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor>     intrinsics;

    /// Extrinsic parameter matrix
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor>     extrinsics;

    /// RGB image
    cv::Mat             image;
};

/**
 * @brief   Struct to store a panorama image taken at a scan position
 * 
 */
struct Panorama
{
    /// Minimum horizontal angle
    float   hmin;

    /// Maximum horizontal angle
    float   hmax;

    /// Vertical field of view
    float   fov;

    /// Vector of camera 
    std::vector<CameraData> images;
};


} // namespace lvr2

#endif /* !CAMDATA_HPP_ */
