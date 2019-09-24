#include "lvr2/registration/OctreeReduction.hpp"

#include <vector>

namespace lvr2
{

OctreeReduction::OctreeReduction(const PointBufferPtr& pointBuffer, const double& voxelSize, const size_t& minPointsPerVoxel)
{

}

OctreeReduction::OctreeReduction(const Vector3f* points, const size_t& n, const double& voxelSize, const size_t& minPointsPerVoxel)
{

}

PointBufferPtr OctreeReduction::getReducedPoints()
{
    return PointBufferPtr(new PointBuffer);
}

void OctreeReduction::getReducedPoints(Vector3f& points, size_t& n)
{

}

} // namespace lvr2