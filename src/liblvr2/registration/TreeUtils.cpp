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
 * TreeUtils.cpp
 *
 *  @date May 16, 2019
 *  @author Malte Hillmann
 */
#include <lvr2/registration/TreeUtils.hpp>

#include <limits>
#include <iostream>
#include <vector>
#include <future>
#include <omp.h>
using namespace std;

namespace lvr2
{

int splitPoints(Vector3f* points, int n, int axis, double splitValue)
{
    int l = 0, r = n - 1;

    while (l < r)
    {
        while (l <= r && points[l](axis) < splitValue)
        {
            l += 1;
        }
        while (r >= l && points[r](axis) >= splitValue)
        {
            if (r == l) // prevent r from going below 0
            {
                break;
            }
            r -= 1;
        }
        if (l < r)
        {
            std::swap(points[l], points[r]);
        }
    }

    return l;
}

void octreeSplit(Vector3f* points, int start, int n, const Vector3f& center, int axis, int* starts, int* counts, int& current);

void createOctree(Vector3f* points, int n, bool* flagged, Vector3f min, Vector3f max, double voxelSize, int maxLeafSize, int threads_left)
{
    if (n <= maxLeafSize)
    {
        return;
    }

    AABB boundingBox(points, n);

    if (boundingBox.difference(boundingBox.longestAxis()) < voxelSize)
    {
        // keep the Point closest to the center
        Vector3f center = boundingBox.avg();
        int closest = 0;
        for (int i = 1; i < n; i++)
        {
            if ((points[i] - center).squaredNorm() < (points[closest] - center).squaredNorm())
            {
                closest = i;
            }
        }
        // flag all other Points for deletion
        for (int i = 0; i < n; i++)
        {
            if (i != closest)
            {
                flagged[i] = true;
            }
        }
        return;
    }

    Vector3f center = (min + max) / 2.0;
    Vector3f dist = (max - min) / 2.0;

    int starts[8] = { 0 };
    int counts[8] = { 0 };
    int current = 0;
    octreeSplit(points, 0, n, center, 0, starts, counts, current);

    int next_threads_left = threads_left;
    if (threads_left > 0)
    {
        int thread_count = 0;
        for (int i = 0; i < 8; i++)
        {
            if (counts[i] > maxLeafSize * 64) // only count significant branches
            {
                thread_count++;
            }
        }

        if (thread_count == 0)
        {
            return;
        }

        next_threads_left = std::max(0, threads_left - thread_count);
    }

    vector<future<void>> futures;

    for (int i = 0; i < 8; i++)
    {
        if (counts[i] <= maxLeafSize)
        {
            continue;
        }

        Vector3f my_min = min;
        Vector3f my_max = max;

        if ((i >> 2) & 1)
        {
            my_min.x() = center.x();
        }
        else
        {
            my_max.x() = center.x();
        }
        if ((i >> 1) & 1)
        {
            my_min.y() = center.y();
        }
        else
        {
            my_max.y() = center.y();
        }
        if (i & 1)
        {
            my_min.z() = center.z();
        }
        else
        {
            my_max.z() = center.z();
        }

        if (threads_left > 0)
        {
            futures.push_back(async(createOctree, points + starts[i], counts[i], flagged + starts[i], my_min, my_max, voxelSize, maxLeafSize, next_threads_left));
        }
        else
        {
            createOctree(points + starts[i], counts[i], flagged + starts[i], my_min, my_max, voxelSize, maxLeafSize, 0);
        }
    }

    for (future<void>& future : futures)
    {
        future.get();
    }
}

void octreeSplit(Vector3f* points, int start, int n, const Vector3f& center, int axis, int* starts, int* counts, int& current)
{
    int split = splitPoints(points + start, n, axis, center[axis]);

    for (int i = 0; i < 2; i++)
    {
        int next_start = start + i * split;
        int next_n = i == 0 ? split : n - split;

        if (axis < 2)
        {
            octreeSplit(points, next_start, next_n, center, axis + 1, starts, counts, current);
        }
        else
        {
            starts[current] = next_start;
            counts[current] = next_n;
            current++;
        }
    }
}

int octreeReduce(Vector3f* points, int n, double voxelSize, int maxLeafSize)
{
    bool* flagged = new bool[n];
    for (int i = 0; i < n; i++)
    {
        flagged[i] = false;
    }

    AABB boundingBox(points, n);

    createOctree(points, n, flagged, boundingBox.min(), boundingBox.max(), voxelSize, maxLeafSize, omp_get_max_threads() * 2);

    // swap_remove all flagged elements
    int i = 0;
    while (i < n)
    {
        if (flagged[i])
        {
            n--;
            if (i == n)
            {
                break;
            }
            std::swap(points[i], points[n]);
            flagged[i] = flagged[n];
        }
        else
        {
            i++;
        }
    }

    delete[] flagged;

    return n;
}


AABB::AABB()
{
    m_min.setConstant(numeric_limits<double>::infinity());
    m_max.setConstant(-numeric_limits<double>::infinity());
    m_sum.setConstant(0.0);
    m_count = 0;
}

AABB::AABB(const Vector3f* points, size_t count)
    : AABB()
{
    for (size_t i = 0; i < count; i++)
    {
        addPoint(points[i]);
    }
}

void AABB::addPoint(const Vector3f& point)
{
    for (int axis = 0; axis < 3; axis++)
    {
        double val = point(axis);
        if (val < m_min(axis))
        {
            m_min(axis) = val;
        }
        if (val > m_max(axis))
        {
            m_max(axis) = val;
        }
    }
    m_sum += point;
    m_count++;
}

int AABB::longestAxis() const
{
    int splitAxis = 0;
    for (int axis = 1; axis < 3; axis++)
    {
        if (difference(axis) > difference(splitAxis))
        {
            splitAxis = axis;
        }
    }
    return splitAxis;
}

} /* namespace lvr2 */
