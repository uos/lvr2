#include "lvr2/registration/AABB.hpp"
#include "lvr2/registration/OctreeReduction.hpp"

#include <vector>

namespace lvr2
{

OctreeReduction::OctreeReduction(PointBufferPtr& pointBuffer, const double& voxelSize, const size_t& minPointsPerVoxel)
{

}

OctreeReduction::OctreeReduction(
    Vector3f* points, 
    const size_t& n0, 
    const double& voxelSize, 
    const size_t& minPointsPerVoxel) : m_voxelSize(voxelSize), m_minPointsPerVoxel(minPointsPerVoxel)
{

    bool* flagged = new bool[n0];
    for (int i = 0; i < n0; i++)
    {
        flagged[i] = false;
    }

    AABB<float> boundingBox(points, n0);

    #pragma omp parallel // allows "pragma omp task"
    #pragma omp single // only execute every task once
    createOctree<Vector3f>(points, n0, flagged, boundingBox.min(), boundingBox.max(), 0);

    // remove all flagged elements
    size_t i = 0;
    size_t n = n0;

    while (i < n)
    {
        if (flagged[i])
        {
            n--;
            if (i == n)
            {
                break;
            }
            points[i] = points[n];
            flagged[i] = flagged[n];
        }
        else
        {
            i++;
        }
    }

    delete[] flagged;
  
}

PointBufferPtr OctreeReduction::getReducedPoints()
{
    return PointBufferPtr(new PointBuffer);
}

void OctreeReduction::getReducedPoints(Vector3f& points, size_t& n)
{

}

} // namespace lvr2