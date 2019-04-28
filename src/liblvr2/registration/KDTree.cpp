
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

#include <lvr2/registration/KDTree.hpp>

namespace lvr2
{

class KDNode : public KDTree
{
public:
    KDNode(int axis, double split, KDTreePtr& lesser, KDTreePtr& greater)
        : axis(axis), split(split), lesser(move(lesser)), greater(move(greater))
    { }

protected:
    virtual void nn_internal(const Vector3d& point, Vector3d*& neighbor, double& maxDist) const override
    {
        double val = point(this->axis);
        //this->lesser->nn_internal(point, neighbor, maxDist);
        //this->greater->nn_internal(point, neighbor, maxDist);
        if (val < this->split)
        {
            this->lesser->nn_internal(point, neighbor, maxDist);
            if (val + maxDist >= this->split)
            {
                this->greater->nn_internal(point, neighbor, maxDist);
            }
        }
        else
        {
            this->greater->nn_internal(point, neighbor, maxDist);
            if (val - maxDist <= this->split)
            {
                this->lesser->nn_internal(point, neighbor, maxDist);
            }
        }
    }

private:
    int axis;
    double split;
    KDTreePtr lesser;
    KDTreePtr greater;
};

class KDLeaf : public KDTree
{
public:
    KDLeaf(Vector3d* points, int count)
        : points(points), count(count)
    { }

protected:
    virtual void nn_internal(const Vector3d& point, Vector3d*& neighbor, double& maxDist) const override
    {
        double maxDistSq = maxDist * maxDist;
        bool changed = false;
        for (int i = 0; i < this->count; i++)
        {
            double dist = (point - this->points[i]).squaredNorm();
            if (dist < maxDistSq)
            {
                neighbor = &this->points[i];
                maxDistSq = dist;
                changed = true;
            }
        }
        if (changed)
        {
            maxDist = sqrt(maxDistSq);
        }
    }

private:
    Vector3d* points;
    int count;
};

struct PointCloudStats
{
    Vector3d min;
    Vector3d max;
    Vector3d avg;
    int count;

    PointCloudStats()
    {
        min.setConstant(INFINITY);
        max.setConstant(-INFINITY);
        avg.setConstant(0);
        count = 0;
    }
    void addPoint(const Vector3d& point)
    {
        for (int axis = 0; axis < 3; axis++)
        {
            double val = point(axis);
            if (val < min(axis))
            {
                min(axis) = val;
            }
            if (val > max(axis))
            {
                max(axis) = val;
            }
            avg(axis) += val;
        }
        count++;
    }
    void finish()
    {
        for (int axis = 0; axis < 3; axis++)
        {
            avg(axis) /= count;
        }
        count = 1;
    }
    double difference(int axis)
    {
        return max(axis) - min(axis);
    }
};

KDTreePtr create_recursive(Vector3d* points, int n, PointCloudStats stats, int maxLeafSize)
{
    if (n <= maxLeafSize)
    {
        return KDTreePtr(new KDLeaf(points, n));
    }
    int split_axis = 0;
    for (int axis = 1; axis < 3; axis++)
    {
        if (stats.difference(axis) > stats.difference(split_axis))
        {
            split_axis = axis;
        }
    }
    double split = stats.avg(split_axis);

    if (stats.difference(split_axis) == 0.0) // all points are exactly the same
    {
        // there is no need to check all of them later on, so just pretend like there is only one
        return KDTreePtr(new KDLeaf(points, 1));
    }

    PointCloudStats lesser_stats;
    PointCloudStats greater_stats;

    int l = 0, r = n - 1;

    while (l < r)
    {
        while (l <= r && points[l](split_axis) < split)
        {
            lesser_stats.addPoint(points[l]);
            l += 1;
        }
        while (r >= l && points[r](split_axis) >= split)
        {
            greater_stats.addPoint(points[r]);
            if (r == l) // prevent r from going below 0
            {
                break;
            }
            r -= 1;
        }
        if (l < r)
        {
            Vector3d tmp = points[l];
            points[l] = points[r];
            points[r] = tmp;
        }
    }

    lesser_stats.finish();
    greater_stats.finish();

    KDTreePtr lesser = create_recursive(points, l, lesser_stats, maxLeafSize);
    KDTreePtr greater = create_recursive(points + l, n - l, greater_stats, maxLeafSize);

    return KDTreePtr(new KDNode(split_axis, split, lesser, greater));
}

KDTreePtr KDTree::create(PointArray points, int n, int maxLeafSize)
{
    PointCloudStats stats;
    for (int i = 0; i < n; i++)
    {
        stats.addPoint(points[i]);
    }
    stats.finish();

    KDTreePtr ret = create_recursive(points.get(), n, stats, maxLeafSize);
    ret->points = points;
    return ret;
}

void KDTree::nearestNeighbor(const Vector3d& point, Vector3d*& neighbor, double& distance, double maxDistance) const
{
    neighbor = nullptr;
    distance = maxDistance;
    nn_internal(point, neighbor, distance);
}

}
