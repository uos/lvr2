
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
#ifndef KDTREE_HPP_
#define KDTREE_HPP_

#include "TreeUtils.hpp"
#include "SLAMScanWrapper.hpp"

#include <memory>
#include <limits>
#include <boost/shared_array.hpp>

namespace lvr2
{

/**
 * @brief a kd-Tree Implementation for nearest Neighbor searches
 */
class KDTree
{
public:
    using PointT = float;
    using Point = Vector3<PointT>;
    using Neighbor = Point*;
    using KDTreePtr = std::shared_ptr<KDTree>;

    /**
     * @brief Creates a new KDTree from the given Scan.
     *
     * @param points        The Point Cloud
     * @param n             The number of points in 'points'
     * @param maxLeafSize   The maximum number of points to use for a Leaf in the Tree
     */
    static std::shared_ptr<KDTree> create(SLAMScanPtr scan, int maxLeafSize = 20);

    /**
     * @brief Finds the nearest neighbor of 'point' that is within 'maxDistance' (defaults to infinity).
     *        The resulting neighbor is written into 'neighbor' (or nullptr if none is found).
     *
     * @param point         The Point whose neighbor is searched
     * @param neighbor      A Pointer that is set to the neighbor or nullptr if none is found
     * @param distance      The final distance between point and neighbor
     * @param maxDistance   The maximum distance allowed between neighbors. Setting this value
     *                      significantly speeds up the search.
     * @return bool true if a neighbors was found, false otherwise
     */
    template<typename T>
    bool nearestNeighbor(
        const Vector3<T>& point,
        Neighbor& neighbor,
        double& distance,
        double maxDistance = std::numeric_limits<double>::infinity()
    ) const
    {
        neighbor = nullptr;
        distance = maxDistance;
        nnInternal(point.template cast<PointT>(), neighbor, distance);

        return neighbor != nullptr;
    }

    virtual ~KDTree() = default;

    /**
     * @brief Finds the nearest neighbors of all points in a Scan using a pre-generated KDTree
     *
     * @param tree          The KDTree to search in
     * @param scan          The Scan to search for
     * @param neighbors     An array to store the results in. neighbors[i] is set to a Pointer to the
     *                      neighbor of points[i] or nullptr if none was found
     * @param maxDistance   The maximum Distance for a Neighbor
     * @param centroid_m    Will be set to the average of all Points in 'neighbors'
     * @param centroid_d    Will be set to the average of all Points in 'points' that have neighbors
     *
     * @return size_t The number of neighbors that were found
     */
    static size_t nearestNeighbors(KDTreePtr tree, SLAMScanPtr scan, Neighbor* neighbors, double maxDistance, Vector3d& centroid_m, Vector3d& centroid_d);

    /**
     * @brief Finds the nearest neighbors of all points in a Scan using a pre-generated KDTree
     *
     * @param tree          The KDTree to search in
     * @param scan          The Scan to search for
     * @param neighbors     An array to store the results in. neighbors[i] is set to a Pointer to the
     *                      neighbor of points[i] or nullptr if none was found
     * @param maxDistance   The maximum Distance for a Neighbor
     *
     * @return size_t The number of neighbors that were found
     */
    static size_t nearestNeighbors(KDTreePtr tree, SLAMScanPtr scan, KDTree::Neighbor* neighbors, double maxDistance);

protected:
    KDTree() = default;
    KDTree(const KDTree&&) = delete;

    virtual void nnInternal(const Point& point, Neighbor& neighbor, double& maxDist) const = 0;

    friend class KDNode;

    boost::shared_array<Point> points;
};

using KDTreePtr = std::shared_ptr<KDTree>;

} /* namespace lvr2 */

#endif /* KDTREE_HPP_ */

