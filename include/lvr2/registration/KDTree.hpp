
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

#include <lvr2/registration/TreeUtils.hpp>

#include <memory>
#include <limits>
#include <boost/shared_array.hpp>

namespace lvr2
{

using PointArray = boost::shared_array<Vector3f>;

class KDTree
{
public:
    /**
     * @brief Creates a new KDTree from the given Point Cloud. Note that this function modifies
     *        the order of elements in 'points'.
     * 
     * @param points        The Point Cloud
     * @param n             The number of points in 'points'
     * @param maxLeafSize   The maximum number of points to use for a Leaf in the Tree
     */
    static std::shared_ptr<KDTree> create(PointArray points, int n, int maxLeafSize = 10);

    /**
     * @brief Creates a new KDTree from the given Point Cloud. Note that this function modifies
     *        the order of elements in 'points'.
     *        This does not take ownership of the pointer. the caller must manage the memory.
     * 
     * @param points        The Point Cloud
     * @param n             The number of points in 'points'
     * @param maxLeafSize   The maximum number of points to use for a Leaf in the Tree
     */
    static std::shared_ptr<KDTree> create(Vector3f* points, int n, int maxLeafSize = 10);

    /**
     * @brief Finds the nearest neighbor of 'point' that is within 'maxDistance' (defaults to infinity).
     *        The resulting neighbor is written into 'neighbor' (or nullptr if none is found).
     * 
     * @param point         The Point whose neighbor is searched
     * @param neighbor      A Pointer that is set to the neighbor or nullptr if none is found
     * @param distance      The final distance between point and neighbor
     * @param maxDistance   The maximum distance allowed between neighbors. Setting this value
     *                      significantly speeds up the search.
     */
    void nearestNeighbor(
        const Vector3f& point,
        Vector3f*& neighbor,
        float& distance,
        float maxDistance = std::numeric_limits<float>::infinity()
    ) const;

    virtual ~KDTree() = default;

protected:
    virtual void nnInternal(const Vector3f& point, Vector3f*& neighbor, float& maxDist) const = 0;

    friend class KDNode;

	PointArray points;
};

using KDTreePtr = std::shared_ptr<KDTree>;

} /* namespace lvr2 */

#endif /* KDTREE_HPP_ */

