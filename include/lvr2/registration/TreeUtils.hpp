
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
 * TreeUtils.hpp
 *
 *  @date May 16, 2019
 *  @author Malte Hillmann
 */
#ifndef TREEUTILS_HPP_
#define TREEUTILS_HPP_

#include <Eigen/Dense>
using Eigen::Matrix4f;
using Eigen::Vector3f;

namespace lvr2
{

/**
 * @brief Sorts a Point array so that all Points smaller than splitValue are on the left
 *
 * Uses the QuickSort Pivot step
 *
 * @param points	 The Point array
 * @param n			 The number of Points
 * @param axis		 The axis to sort by
 * @param splitValue The value to sort by
 *
 * @returns The number of smaller elements. points + this value gives the start of the greater elements
 */
int splitPoints(Vector3f* points, int n, int axis, float splitValue);

/**
 * @brief Reduces a Point Cloud using an Octree with a minimum Voxel size
 *
 * @param points    The Point Cloud
 * @param n		    The number of Points in the Point Cloud
 * @param voxelSize The minimum size of a Voxel
 *
 * @returns the new number of Points in the Point Cloud
 */
int octreeReduce(Vector3f* points, int n, float voxelSize, int maxLeafSize = 10);


/**
 * @brief A struct to calculate the Axis Aligned Bounding Box and Average Point of a Point Cloud
 */
class AABB
{
    Vector3f m_min;
    Vector3f m_max;
    Vector3f m_sum;
    size_t m_count;

public:
    AABB();
    AABB(const Vector3f* points, size_t count);

    /// Returns the "lower left" Corner of the Bounding Box, as in the smallest x, y, z of the Point Cloud.
    const Vector3f& min() const
    {
        return m_min;
    }

    /// Returns the "upper right" Corner of the Bounding Box, as in the largest x, y, z of the Point Cloud.
    const Vector3f& max() const
    {
        return m_max;
    }

    /// Returns the average of all the Points in the Point Cloud.
    Vector3f avg() const
    {
        return m_sum / m_count;
    }

    /// Returns the number of Points in the Point Cloud
    size_t count() const
    {
        return m_count;
    }

    /// adds a Point to the Point Cloud
    void addPoint(const Vector3f& point);

    /// Returns the smallest value of an axis of the Point Cloud.
    float min(int axis) const
    {
        return m_min[axis];
    }

    ///	Returns the largest value of an axis of the Point Cloud.
    float max(int axis) const
    {
        return m_max[axis];
    }

    /// Returns the average of an axis of all the Points in the Point Cloud.
    float avg(int axis) const
    {
        return m_sum[axis] / m_count;
    }

    /// Calculates the size of the Bounding Box along a certain axis
    float difference(int axis) const
    {
        return max(axis) - min(axis);
    }

    /// Calculates the axis that has the largest size of the Bounding Box
    int longestAxis() const;
};

} /* namespace lvr2 */

#endif /* TREEUTILS_HPP_ */

