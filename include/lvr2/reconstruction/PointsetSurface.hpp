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
 * PointsetSurface.hpp
 *
 *  @date 24.01.2012
 *  @author Thomas Wiemann
 */

#ifndef LVR2_RECONSTRUCTION_POINTSETSURFACE_HPP_
#define LVR2_RECONSTRUCTION_POINTSETSURFACE_HPP_

#include <memory>
#include <utility>

#include "lvr2/geometry/Normal.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/reconstruction/SearchTree.hpp"

using std::pair;

namespace lvr2
{


/**
 * @brief       An interface class to wrap all functionality that is needed
 *              to generate a surface approximation from point cloud data.
 *
 *              Classes that implement this interface can be used for Marching
 *              Cubes based mesh generation algorithms in via
 *              the @ref SurfaceReconstruction interface.
 */
template<typename BaseVecT>
class PointsetSurface
{
public:

    /**
     * @brief Returns the distance of vertex v from the nearest tangent plane
     *
     * @param p A grid point
     *
     * @return The first value is the projected distance from the query point
     *         to the isosurface. The second value is the euclidian distance to
     *         the nearest data point.
     */
    virtual pair<typename BaseVecT::CoordType, typename BaseVecT::CoordType>
        distance(BaseVecT v) const = 0;
    /**
     * @brief   Calculates surface normals for each data point in the given
     *          PointBuffeer. If the buffer alreay contains normal information
     *          it will be overwritten with the new normals.
     */
    virtual void calculateSurfaceNormals() = 0;



    /**
     * @brief   Interpolates a surface normal at the given position
     * @param   position the position to calculate a normal for
     *
     * @return  The normal
     */
    virtual Normal<float> getInterpolatedNormal(const BaseVecT& position) const;

    /**
     * @brief   Returns the internal point buffer. After a call of
     *          @ref calculateSurfaceNormals the buffer will contain
     *          normal information.
     */
    virtual PointBufferPtr pointBuffer() const;

    /**
     * @brief   Returns a pointer to the search tree
     */
    std::shared_ptr<SearchTree<BaseVecT>> searchTree() const;

    /**
     * @brief   Returns the bounding box of the point set
     */
    const BoundingBox<BaseVecT>& getBoundingBox() const;

    /**
     * @brief   If k is > 0, each normal will be averaged with its k
     *          neighbors.
     */
    void setKi(int k) { m_ki = k; }

    /**
     * @brief   Sets the size of the k-neighborhood that is used for
     *          normal estimation.
     */
    void setKn(int k) { m_kn = k; }

    /**
     * @brief   Sets the number of points that is used for distance
     *          evaluation, i.e. an average of the distance to the
     *          k nearest data points is given (useful in noisy
     *          data sets).
     */
    void setKd(int k) { m_kd = k; }

protected:

    /**
     * @brief   Constructor. Stores the given buffer pointer. If the point
     *          buffer does not contain surface normals, you will have to call
     *          @ref calculateSurfaceNormals before the first call @distance.
     */
    PointsetSurface(PointBufferPtr pointcloud);

    PointsetSurface() {};

    /// The point cloud used for surface approximation
    PointBufferPtr m_pointBuffer;

    /// The search tree that is built from the point cloud data
    std::shared_ptr<SearchTree<BaseVecT>> m_searchTree;

    /// The bounding box of the point cloud
    BoundingBox<BaseVecT> m_boundingBox;

    /// The number of points used for normal estimation
    int m_kn;

    /// The number of points used for normal interpolation
    int m_ki;

    /// The number of points used for distance function evaluation
    int m_kd;
};

template <typename BaseVecT>
using PointsetSurfacePtr = std::shared_ptr<PointsetSurface<BaseVecT>>;

} // namespace lvr2

#include "PointsetSurface.tcc"

#endif /* LVR2_RECONSTRUCTION_POINTSETSURFACE_HPP_ */
