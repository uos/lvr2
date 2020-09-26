#include <iostream>
#include <memory>
#include <tuple>
#include <stdlib.h>

#include <boost/optional.hpp>

// lvr2 includes
#include "lvr2/registration/TransformUtils.hpp"
#include "lvr2/types/MatrixTypes.hpp"

int main(int argc, char** argv)
{
    std::cout << "Coordinate Example" << std::endl;
    // Overview Coordinate Systems

    /** LVR / ROS
    *        z  x    
    *        | /
    *   y ___|/ 
    * 
    *  - x: front
    *  - y: left
    *  - z: up
    *  - scale: m
    */

    /** OpenCV
     * 
     *        z    
     *       /
     *      /___ x
     *     |
     *     |
     *     y
     * 
     * - x: right
     * - y: down
     * - z: front
     * - scale: m
     */


    // x: 2.0, y: 0.5, z: 1.0
    // front: 1.0, right: 2.0, down: 0.5
    lvr2::Vector3d cv_point = {2.0, 0.5, 1.0};
    std::cout << "cv point: " << cv_point.transpose() << std::endl;

    // convert to lvr
    // should be x(front): 1.0, y(left): -2.0, z(up): -0.5 
    lvr2::Vector3d lvr_point = lvr2::openCvToLvr(cv_point);
    std::cout << "lvr point: " << lvr_point.transpose() << std::endl;

    if(lvr2::lvrToOpenCv(lvr_point) == cv_point)
    {
        std::cout << "LVR <-> OpenCV - Point: Success" << std::endl;
    }

    // check opencv transformation

    lvr2::Rotationd cv_rot, lvr_rot;

    double roll = -0.25*M_PI; // cv: z, lvr: x
    double pitch = 1.6*M_PI; // cv: -x, lvr: y
    double yaw = -0.06*M_PI; // cv: -y, lvr: z

    // cv rotate x: pitch
    cv_rot = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitZ())
        * Eigen::AngleAxisd(-pitch, Eigen::Vector3d::UnitX())
        * Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitY());
    
    // cv -> lvr
    lvr_rot = lvr2::openCvToLvr(cv_rot);


    // cv_rot 

    lvr2::Vector3d cv_point_rotated = cv_rot * cv_point;
    lvr2::Vector3d lvr_point_rotated = lvr_rot * lvr_point;

    // lvr2::openCvToLvr(cv_point_rotated) - lvr_point_rotated
    if((lvr2::openCvToLvr(cv_point_rotated) - lvr_point_rotated).norm() < 0.000001)
    {
        std::cout << "LVR <-> OpenCV - Rotation Matrix: Success" << std::endl;
    } else {
        std::cout << "LVR <-> OpenCV - Rotation Matrix: Wrong" << std::endl;
    }

    // transformation
    lvr2::Transformd cv_transform, lvr_transform;

    cv_transform = lvr2::Transformd::Identity();
    cv_transform.block<3,3>(0,0) = cv_rot;
    cv_transform(0,2) = 2.0;
    cv_transform(1,2) = 5.0;
    cv_transform(2,2) = -1.0;

    lvr_transform = lvr2::openCvToLvr(cv_transform);

    lvr2::Vector3d cv_point_transformed = cv_transform * cv_point;
    lvr2::Vector3d lvr_point_transformed = lvr_transform * lvr_point;

    if((lvr2::openCvToLvr(cv_point_transformed)-lvr_point_transformed).norm() < 0.000001)
    {
        std::cout << "LVR <-> OpenCV - Transformation Matrix: Success" << std::endl;
    } else {
        std::cout << "LVR <-> OpenCV - Transformation Matrix: Wrong" << std::endl;
    }

    return 0;
}