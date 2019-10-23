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
 * TreeUtils.tcc
 *
 *  @date May 16, 2019
 *  @author Malte Hillmann
 *  @author Thomas Wiemann
 */

#include "lvr2/registration/TreeUtils.hpp"
#include "lvr2/registration/AABB.hpp"

#include <limits>
#include <vector>
using namespace std;
using namespace Eigen;

namespace lvr2
{

int splitPoints(Vector3f* points, int n, int axis, double splitValue)
{
    int l = 0, r = n - 1;

    while (l < r)
    {
        while (l < r && points[l](axis) < splitValue)
        {
            ++l;
        }
        while (r > l && points[r](axis) >= splitValue)
        {
            --r;
        }
        if (l < r)
        {
            std::swap(points[l], points[r]);
        }
    }

    return l;
}

void createOctree(Vector3f* points,
                  int n,
                  bool* flagged,
                  const Vector3f& min,
                  const Vector3f& max,
                  int level,
                  double voxelSize,
                  int maxLeafSize)
{
    if (n <= maxLeafSize)
    {
        return;
    }

    int axis = level % 3;
    Vector3f center = (max + min) / 2.0;

    if (max[axis] - min[axis] <= voxelSize)
    {
        // keep the Point closest to the center
        int closest = 0;
        double minDist = (points[closest] - center).squaredNorm();
        for (int i = 1; i < n; i++)
        {
            double dist = (points[i] - center).squaredNorm();
            if (dist < minDist)
            {
                closest = i;
                minDist = dist;
            }
        }
        // flag all other Points for deletion
        for (int i = 0; i < n; i++)
        {
            flagged[i] = i != closest;
        }
        return;
    }

    int l = splitPoints(points, n, axis, center[axis]);

    Vector3f lMin = min, lMax = max;
    Vector3f rMin = min, rMax = max;

    lMax[axis] = center[axis];
    rMin[axis] = center[axis];

    if (l > maxLeafSize)
    {
        #pragma omp task
        createOctree(points    , l    , flagged    , lMin, lMax, level + 1, voxelSize, maxLeafSize);
    }

    if (n - l > maxLeafSize)
    {
        #pragma omp task
        createOctree(points + l, n - l, flagged + l, rMin, rMax, level + 1, voxelSize, maxLeafSize);
    }
}

int octreeReduce(Vector3f* points, int n, double voxelSize, int maxLeafSize)
{
    bool* flagged = new bool[n];
    for (int i = 0; i < n; i++)
    {
        flagged[i] = false;
    }

    AABB<float> boundingBox(points, (size_t)n);

    #pragma omp parallel // allows "pragma omp task"
    #pragma omp single // only execute every task once
    createOctree(points, n, flagged, boundingBox.min(), boundingBox.max(), 0, voxelSize, maxLeafSize);

    // remove all flagged elements
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
            points[i] = points[n];
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

} /* namespace lvr2 */
