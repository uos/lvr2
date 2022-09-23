#ifndef FPFH
#define FPFH

#include <Eigen/Core>

#include <memory>

#include <lvr2/reconstruction/SearchTree.hpp>
#include <lvr2/types/PointBuffer.hpp>

namespace lvr2
{

using FPFHFeature = Eigen::MatrixXf;
using FPFHFeaturePtr = std::shared_ptr<FPFHFeature>;

/**
 * @brief Computes FPFH features for the given point cloud. Requires 
 *        point normals in the passed PointBufferPtr.
 * 
 *        Implementation based on the feature computation in Open3D:
 *        http://www.open3d.org/docs/release/cpp_api/_feature_8cpp.html
 * 
 * @param pointCloud        A point cloud containing normals
 * @param k                 Number of nearest neighbors
 * @return FPFHFeaturePtr   An Eigen matrix containing the computed features
 *                          (33 x numPoints)
 */
FPFHFeaturePtr computeFPFHFeatures(const PointBufferPtr pointCloud, size_t k);

} // namespace lvr2

#endif // FPFH
