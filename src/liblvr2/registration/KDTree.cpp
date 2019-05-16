
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
 * KDTree.hpp
 *
 *  @date Apr 28, 2019
 *  @author Malte Hillmann
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
    virtual void nnInternal(const Vector3d& point, Vector3d*& neighbor, double& maxDist) const override
    {
        double val = point(this->axis);
        if (val < this->split)
        {
            this->lesser->nnInternal(point, neighbor, maxDist);
            if (val + maxDist >= this->split)
            {
                this->greater->nnInternal(point, neighbor, maxDist);
            }
        }
        else
        {
            this->greater->nnInternal(point, neighbor, maxDist);
            if (val - maxDist <= this->split)
            {
                this->lesser->nnInternal(point, neighbor, maxDist);
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
    virtual void nnInternal(const Vector3d& point, Vector3d*& neighbor, double& maxDist) const override
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

KDTreePtr create_recursive(Vector3d* points, int n, int maxLeafSize)
{
    if (n <= maxLeafSize)
    {
        return KDTreePtr(new KDLeaf(points, n));
    }

    AABB boundingBox(points, n);

    int splitAxis = boundingBox.longestAxis();
    double splitValue = boundingBox.avg(splitAxis);

    if (boundingBox.difference(splitAxis) == 0.0) // all points are exactly the same
    {
        // there is no need to check all of them later on, so just pretend like there is only one
        return KDTreePtr(new KDLeaf(points, 1));
    }

    int l = splitPoints(points, n, splitAxis, splitValue);

    KDTreePtr lesser = create_recursive(points, l, maxLeafSize);
    KDTreePtr greater = create_recursive(points + l, n - l, maxLeafSize);

    return KDTreePtr(new KDNode(splitAxis, splitValue, lesser, greater));
}

KDTreePtr KDTree::create(PointArray points, int n, int maxLeafSize)
{
    KDTreePtr ret = create_recursive(points.get(), n, maxLeafSize);
    ret->points = points;
    return ret;
}

void KDTree::nearestNeighbor(const Vector3d& point, Vector3d*& neighbor, double& distance, double maxDistance) const
{
    neighbor = nullptr;
    distance = maxDistance;
    nnInternal(point, neighbor, distance);
}

}
