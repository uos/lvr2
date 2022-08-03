
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
#pragma once

#include "lvr2/reconstruction/SearchTree.hpp"

#include <memory>
#include <limits>
#include <queue>

namespace lvr2
{

/**
 * @brief a kd-Tree Implementation for nearest Neighbor searches
 *
 * Can contain any type of data, as long as it implements  <float/double> operator[](uint axis)  for axis in [0, N).
 * Specifically, one could create a type like:
 * struct PointWithNormal {
 *     Vector3f point;
 *     Vector3f normal;
 *     float operator[](uint i) const { return point[i]; }
 * };
 * Or
 * struct PointWithHandle {
 *     Vector3f point;
 *     VertexHandle handle;
 *     float operator[](uint i) const { return point[i]; }
 * };
 * To have associated metadata with the points, that can then be obtained from the neighbors.
 *
 * @tparam PointT The type of the data to be stored in the tree
 * @tparam N The dimensions of PointT
 */
template<typename PointT, unsigned int N = 3>
class KDTree
{
    static_assert(N > 0, "KDTree: N must be greater than 0");
public:
    using Ptr = std::shared_ptr<KDTree<PointT, N>>;

    /**
     * @brief Construct a new KDTree.
     *
     * @param points The points to be stored in the tree.
     * @param numPoints The number of elements in points.
     * @param maxLeafSize The maximum number of points per leaf.
     */
    static Ptr create(std::unique_ptr<PointT[]>&& points, size_t numPoints, size_t maxLeafSize = 20);

    virtual ~KDTree() = default;

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
    template<typename InPointT, typename FloatT>
    bool nnSearch(const InPointT& point,
                  PointT*& neighbor,
                  FloatT& distance,
                  double maxDistance = std::numeric_limits<double>::infinity()) const;

    /**
     * @brief Finds the 'k' nearest neighbor of 'point' that are within 'maxDistance' (defaults to infinity).
     *        The resulting neighbor is written into 'neighbor' (or nullptr if none is found).
     *
     * @param point         The Point whose neighbor is searched
     * @param k             The number of neighbors to find
     * @param neighbors     Will be filled with the neighbors
     * @param distances     Will be filled with the distances to the neighbors
     * @param maxDistance   The maximum distance allowed between neighbors. Setting this value
     *                      significantly speeds up the search.
     * @return size_t The number of neighbors found (Usually k)
     */
    template<typename InPointT, typename FloatT>
    size_t knnSearch(const InPointT& point,
                     size_t k,
                     std::vector<PointT*>& neighbors,
                     std::vector<FloatT>& distances,
                     double maxDistance = std::numeric_limits<double>::infinity()) const;

    /**
     * @brief Same as knnSearch, but does not gather distances.
     */
    template<typename InPointT>
    size_t knnSearch(const InPointT& point,
                     size_t k,
                     std::vector<PointT*>& neighbors,
                     double maxDistance = std::numeric_limits<double>::infinity()) const;

    /**
     * @brief Returns the number of points in the tree.
     */
    size_t numPoint() const
    {
        return m_numPoints;
    }

    /**
     * @brief Returns the points stored in the tree.
     *
     * Note that the order is different than the one passed to the constructor.
     */
    const PointT* points() const
    {
        return m_points.get();
    }

protected:
    KDTree(std::unique_ptr<PointT[]>&& points, size_t numPoints)
        : m_points(std::move(points)), m_numPoints(numPoints)
    {}
    KDTree(PointT* points, size_t numPoints)
        : m_points(points), m_numPoints(numPoints)
    {}

    /// A Point with its squared distance to the query Point for easy comparison
    struct DistPoint
    {
        PointT* point;
        double distanceSq;
        bool operator<(const DistPoint& other) const
        {
            return distanceSq < other.distanceSq;
        }
    };

    /// A Priority queue to keep track of the k nearest Points
    using Queue = std::priority_queue<DistPoint>;

    using QueryPoint = Eigen::Matrix<double, N, 1>;
    template<typename InPointT>
    QueryPoint toQueryPoint(const InPointT& point) const
    {
        QueryPoint qp;
        for (unsigned int i = 0; i < N; i++)
        {
            qp[i] = point[i];
        }
        return qp;
    }

    class KDTreeInternal
    {
    public:
        /**
         * @brief Internal recursive version of nnSearch. Provided by subclasses
         *
         * @param point The query Point to search around
         * @param neighbor The current nearest neighbor or nullptr if none is found
         * @param worstDistSq The remaining search radius for new Neighbors (squared)
         */
        virtual void nnInternal(const QueryPoint& point, DistPoint& neighbor) const = 0;

        /**
         * @brief Internal recursive version of knnSearch. Provided by subclasses
         *
         * @param point The query Point to search around
         * @param neighbors The Queue to place the Neighbors into
         * @param worstDistSq The remaining search radius for new Neighbors (squared)
         * @param k The number of Neighbors to find
         */
        virtual void knnInternal(const QueryPoint& point, size_t k, Queue& neighbors, double& worstDistSq) const = 0;

        virtual ~KDTreeInternal() = default;
    };

    using KDPtr = std::unique_ptr<KDTreeInternal>;

    class KDNode;
    class KDLeaf;
    void init(size_t maxLeafSize = 20);
    KDPtr createRecursive(PointT* start, PointT* end, const QueryPoint& min, const QueryPoint& max, size_t maxLeafSize);

    size_t m_numPoints;
    std::unique_ptr<PointT[]> m_points;
    std::unique_ptr<KDTreeInternal> m_tree;
};

template<typename PointT, unsigned int N = 3>
using KDTreePtr = typename KDTree<PointT, N>::Ptr;


template<typename BaseVecT>
struct IndexedPoint
{
    BaseVecT point;
    size_t index;
    typename BaseVecT::CoordType operator[](unsigned int i) const
    {
        return point[i];
    }
};

template<typename BaseVecT>
class SearchKDTree : public KDTree<IndexedPoint<BaseVecT>>, public SearchTree<BaseVecT>
{
    using CoordT = typename BaseVecT::CoordType;
    using PointT = IndexedPoint<BaseVecT>;
public:
    SearchKDTree(PointBufferPtr buffer);

    int kSearch(
        const BaseVecT& qp,
        int k,
        std::vector<size_t>& indices,
        std::vector<CoordT>& distances
    ) const override;

    int kSearch(
        const BaseVecT& qp,
        int k,
        std::vector<size_t>& indices
    ) const override;

    int radiusSearch(
        const BaseVecT& qp,
        int k,
        float r,
        std::vector<size_t>& indices,
        std::vector<CoordT>& distances
    ) const override;
};

} // namespace lvr2

#include "KDTree.tcc"
