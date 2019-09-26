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
 * EigenSVDPointAlign.hpp
 *
 *  @date Sep 24, 2019
 * 
 *  @author Malte Hillmann
 *  @author Thomas Wiemann
 */

#include "lvr2/types/MatrixTypes.hpp"

#include <limits>
#include <iostream>

namespace lvr2
{

/**
 * @brief A struct to calculate the Axis Aligned Bounding Box and Average Point of a Point Cloud
 */
template<typename T>
class AABB
{
    Vector3<T> m_min;
    Vector3<T> m_max;
    Vector3<T> m_sum;
    size_t m_count;

public:
    AABB();

    /**
     * @brief Construct a new AABB object
     * 
     * @tparam P        Array of point representations that support []-based acces to coordinates
     * @param points    Array of points
     * @param count     Number of points in the point array
     */
    template<typename P>
    AABB(P* points, size_t count) : AABB()
    {
        for (size_t i = 0; i < count; i++)
        {
            addPoint(points[i]);
        }
    }

    /**
     * @brief Construct a new AABB object
     * 
     * @tparam P        Set of points. Points are accessed via the [] operator. The returned 
     *                  point types must support [] operater access to the coordinates. LVR2's
     *                  channel object suppoort these requirements
     * @param points    Set of point
     * @param count     Number of points
     */
    template<typename P>
    AABB(P& points, size_t count) : AABB()
    {
        for(size_t i = 0; i < count; i++)
        {
            addPoint(points[i]);
        }
    }

    /// Returns the "lower left" Corner of the Bounding Box, as in the smallest x, y, z of the Point Cloud.
    const Vector3<T>& min() const;

    /// Returns the "upper right" Corner of the Bounding Box, as in the largest x, y, z of the Point Cloud.
    const Vector3<T>& max() const;

    /// Returns the average of all the Points in the Point Cloud.
    Vector3<T> avg() const;

    /// Returns the number of Points in the Point Cloud
    size_t count() const;

    /// adds a Point to the Point Cloud
    template<typename P>
    void addPoint(const P& point)
    {
        for (int axis = 0; axis < 3; axis++)
        {
            double val = point[axis];
            if (val < m_min[axis])
            {
                m_min[axis] = val;
            }
            if (val > m_max[axis])
            {
                m_max[axis] = val;
            }
            m_sum[axis] += val;
        }
        m_count++;
    }

    /// Calculates the size of the Bounding Box along a certain axis
    T difference(int axis) const;

    /// Calculates the axis that has the largest size of the Bounding Box
    int longestAxis() const;
};

} // namespace lvr2

#include "lvr2/registration/AABB.tcc"