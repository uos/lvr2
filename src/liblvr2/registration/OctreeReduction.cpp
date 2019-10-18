#include "lvr2/registration/AABB.hpp"
#include "lvr2/registration/OctreeReduction.hpp"
#include "lvr2/io/IOUtils.hpp"

#include <vector>

namespace lvr2
{

OctreeReduction::OctreeReduction(
    PointBufferPtr &pointBuffer,
    const double &voxelSize,
    const size_t &minPointsPerVoxel)
    : m_voxelSize(voxelSize),
      m_minPointsPerVoxel(minPointsPerVoxel),
      m_numPoints(pointBuffer->numPoints()),
      m_pointBuffer(pointBuffer)
{
    size_t n = pointBuffer->numPoints();
    m_flags = new bool[n];
    for (int i = 0; i < n; i++)
    {
        m_flags[i] = false;
    }

    typename lvr2::Channel<float>::Optional pts_opt = pointBuffer->getChannel<float>("points");
    if(pts_opt)
    {
        lvr2::Channel<float> points = *pts_opt;
        AABB<float> boundingBox(points, n);

        #pragma omp parallel // allows "pragma omp task"
        #pragma omp single   // only execute every task once
        createOctree(pointBuffer, 0, n, m_flags, boundingBox.min(), boundingBox.max(), 0);
    }
}

OctreeReduction::OctreeReduction(
    Vector3f *points,
    const size_t &n0,
    const double &voxelSize,
    const size_t &minPointsPerVoxel) : m_voxelSize(voxelSize), m_minPointsPerVoxel(minPointsPerVoxel)
{

    m_flags = new bool[n0];
    for (int i = 0; i < n0; i++)
    {
        m_flags[i] = false;
    }

    AABB<float> boundingBox(points, n0);

#pragma omp parallel // allows "pragma omp task"
#pragma omp single   // only execute every task once
    createOctree<Vector3f>(points, n0, m_flags, boundingBox.min(), boundingBox.max(), 0);
}

PointBufferPtr OctreeReduction::getReducedPoints()
{
    std::vector<size_t> reducedIndices;
    for (size_t i = 0; i < m_numPoints; i++)
    {
        if (!m_flags[i])
        {
            reducedIndices.push_back(i);
        }
    }

    return subSamplePointBuffer(m_pointBuffer, reducedIndices);
}

void OctreeReduction::getReducedPoints(Vector3f &points, size_t &n)
{
}

} // namespace lvr2