#ifndef __CAMERAMODEL_HPP__
#define __CAMERAMODEL_HPP__

#include <memory>

#include <Eigen/Dense>

#include "lvr2/types/MatrixTypes.hpp"

namespace lvr2
{

template<typename T>
struct PinholeModel
{
    double fx = 0;
    double fy = 0;
    double cx = 0;
    double cy = 0;
    unsigned width = 0;
    unsigned height = 0;
    std::vector<T> k;
    std::string distortionModel = "unknown";
};

template<typename T>
using PinholeModelPtr = std::shared_ptr<PinholeModel<T>>;

using PinholeModeld = PinholeModel<double>;
using PinholeModelf = PinholeModel<float>;


} // namespace lvr2

#endif