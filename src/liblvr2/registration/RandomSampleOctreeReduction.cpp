/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * OctreeReduction.cpp
 *
 * @date Sep 23, 2019
 * @author Malte Hillmann
 * @author Thomas Wiemann
 * @author Justus Braun
 */

#include "lvr2/registration/OctreeReduction.hpp"
#include "lvr2/geometry/pmp/BoundingBox.h"
#include "lvr2/util/IOUtils.hpp"

#include <random>

namespace lvr2
{

RandomSampleOctreeReduction::RandomSampleOctreeReduction(PointBufferPtr pointBuffer, float voxelSize, size_t maxPointsPerVoxel)
    : OctreeReductionBase(pointBuffer, voxelSize, maxPointsPerVoxel)
{
    if (m_numPoints == 0)
    {
        return;
    }
    auto pts_opt = pointBuffer->getChannel<float>("points");
    if (pts_opt)
    {
        auto pts = *pts_opt;
        m_points = new Vector3f[m_numPoints];
        m_pointIndices.resize(m_numPoints);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < m_numPoints; i++)
        {
            m_points[i] << pts[i][0], pts[i][1], pts[i][2];
            m_pointIndices[i] = i;
        }

        init();

        m_pointIndices.resize(m_numPoints);
        m_pointIndices.shrink_to_fit();
        delete[] m_points;
        m_points = nullptr;
    }
}

RandomSampleOctreeReduction::RandomSampleOctreeReduction(Vector3f* points, size_t& n, float voxelSize, size_t maxPointsPerVoxel)
    :  OctreeReductionBase(nullptr, voxelSize, maxPointsPerVoxel), m_points(points)
{
    if (m_numPoints == 0)
    {
        return;
    }
    init();
    n = m_numPoints;
}

void RandomSampleOctreeReduction::init()
{
    m_flags = new bool[m_numPoints];
    std::fill_n(m_flags, m_numPoints, false);

    pmp::BoundingBox bb;
    #pragma omp parallel for schedule(static) reduction(+:bb)
    for (size_t i = 0; i < m_numPoints; i++)
    {
        bb += m_points[i];
    }

    // make the bounding box a perfect cube
    {
        float longestAxis = bb.longest_axis_size();
        pmp::Point halfSize = pmp::Point::Constant(longestAxis / 2.0f);
        pmp::Point center = bb.center();
        bb = pmp::BoundingBox(center - halfSize, center + halfSize);
    }

    #pragma omp parallel // allows "pragma omp task"
    #pragma omp single   // only execute every task once
    createOctree(0, m_numPoints, bb.min(), bb.max());

    size_t l = 0, r = m_numPoints - 1;
    for (;;)
    {
        // find first deleted and last un-deleted
        while (l < r && !m_flags[l])
        {
            l++;
        }
        while (l < r && m_flags[r])
        {
            r--;
        }
        if (l >= r)
        {
            break;
        }
        if (m_pointBuffer)
        {
            std::swap(m_pointIndices[l], m_pointIndices[r]);
        }
        else
        {
            std::swap(m_points[l], m_points[r]);
        }
        l++;
        r--;
    }
    m_numPoints = l;

    delete[] m_flags;
    m_flags = nullptr;
}

PointBufferPtr RandomSampleOctreeReduction::getReducedPoints()
{
    if (!m_pointBuffer)
    {
        throw std::runtime_error("OctreeReduction::getReducedPoints can only be called if a PointBuffer was passed to the constructor.");
    }
    return subSamplePointBuffer(m_pointBuffer, m_pointIndices);
}

void RandomSampleOctreeReduction::createOctree(size_t start, size_t n, const Vector3f& min, const Vector3f& max, unsigned int level)
{
    // Stop recursion - not enough points in voxel
    if (n <= m_maxPointsPerVoxel)
    {
        return;
    }

    // Determine split axis and compute new center
    unsigned int axis = level % 3;
    Vector3f center = (max + min) / 2.0;

    // Stop recursion if voxel size is below given limit
    if (max[axis] - min[axis] <= m_voxelSize)
    {
        // Mark all points in voxel as deleted and then un-mark the ones to keep
        std::fill_n(m_flags + start, n, true);

        static thread_local std::mt19937 randomEngine; // one random engine per omp thread
        static thread_local bool engineInitialized = false;
        if (!engineInitialized)
        {
            randomEngine.seed(std::random_device()());
            engineInitialized = true;
        }

        std::uniform_int_distribution<int> dist(start, start + n - 1);
        for (size_t i = 0; i < m_maxPointsPerVoxel; i++)
        {
            // Randomly select points to keep. This may select the same point multiple times, but that's fine.
            m_flags[dist(randomEngine)] = false;
        }
        return;
    }

    // Sort and get new split index
    size_t startRight = splitPoints(start, n, axis, center[axis]);

    Vector3f lMin = min, lMax = max;
    Vector3f rMin = min, rMax = max;

    lMax[axis] = center[axis];
    rMin[axis] = center[axis];

    size_t numPointsLeft = startRight - start;
    size_t numPointsRight = (start + n) - startRight;
    bool leftSplit = numPointsLeft > m_maxPointsPerVoxel;
    bool rightSplit = numPointsRight > m_maxPointsPerVoxel;

    if (leftSplit && rightSplit)
    {
        // both trees needed => spawn left as task, do right on this thread
        #pragma omp task
        createOctree(start,      numPointsLeft,  lMin, lMax, level + 1);

        createOctree(startRight, numPointsRight, rMin, rMax, level + 1);
    }
    else if (leftSplit)
    {
        createOctree(start,      numPointsLeft,  lMin, lMax, level + 1);
    }
    else if (rightSplit)
    {
        createOctree(startRight, numPointsRight, rMin, rMax, level + 1);
    }
}

size_t RandomSampleOctreeReduction::splitPoints(size_t start, size_t n, unsigned int axis, float splitValue)
{
    size_t l = start;
    size_t r = start + n - 1;

    for (;;)
    {
        // find first deleted and last un-deleted
        while (l < r && m_points[l][axis] < splitValue)
        {
            ++l;
        }
        while (l < r && m_points[r][axis] >= splitValue)
        {
            --r;
        }

        if (l >= r)
        {
            break;
        }

        std::swap(m_points[l], m_points[r]);
        if (m_pointBuffer)
        {
            std::swap(m_pointIndices[l], m_pointIndices[r]);
        }
    }
    return l;
}

} // namespace lvr2