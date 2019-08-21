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
 *  @date Feb 21, 2014
 *  @author Thomas Wiemann
 */
#ifndef EIGENSVDPOINTALIGN_HPP_
#define EIGENSVDPOINTALIGN_HPP_

#include "SLAMScanWrapper.hpp"
#include "lvr2/types/MatrixTypes.hpp"

#include <Eigen/Dense>

namespace lvr2
{

template<typename T, typename PointT = float>
class EigenSVDPointAlign
{
public:
    using Vec3 = Vector3<T>;
    using Mat4 = Transform<T>;
    using Mat3 = Eigen::Matrix<T, 3, 3>;
    using Point3 = Vector3<PointT>;
    using PointPairVector = std::vector<std::pair<Point3, Point3>>;

    EigenSVDPointAlign() {};

    /**
     * @brief Calculates the estimated Transformation to match a Data Pointcloud to a Model
     *        Pointcloud
     * 
     * Apply the resulting Transform to the Data Pointcloud.
     *
     * @param scan       The Data Pointcloud
     * @param neighbors  An array containing a Pointer to a neighbor in the Model Pointcloud for
     *                   each Point in `scan`, or nullptr if there is no neighbor for a Point
     * @param centroid_m The center of the Model Pointcloud
     * @param centroid_d The center of the Data Pointcloud
     * @param align      Will be set to the Transformation
     *
     * @return The average Point-to-Point error of the Scans
     */
    T alignPoints(
        SLAMScanPtr scan,
        Point3** neighbors,
        const Vec3& centroid_m,
        const Vec3& centroid_d,
        Mat4& align) const;

    /**
     * @brief Calculates the estimated Transformation to match a Data Pointcloud to a Model
     *        Pointcloud
     * 
     * Apply the resulting Transform to the Data Pointcloud.
     *
     * @param points     A vector of pairs with (model, data) Points
     * @param centroid_m The center of the Model Pointcloud
     * @param centroid_d The center of the Data Pointcloud
     * @param align      Will be set to the Transformation
     *
     * @return The average Point-to-Point error of the Scans
     */
    T alignPoints(
        PointPairVector& points,
        const Vec3& centroid_m,
        const Vec3& centroid_d,
        Mat4& align) const;
};

} /* namespace lvr2 */

#include "EigenSVDPointAlign.tcc"

#endif /* EIGENSVDPOINTALIGN_HPP_ */
