/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
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

#include <lvr2/io/PointBuffer.hpp>
#include <lvr2/geometry/BoundingBox.hpp>
#include <lvr2/reconstruction/SearchTree.hpp>

using std::pair;

namespace lvr2
{

/**
 * @brief       An interface class to wrap all functionality that is needed
 *              to generate a surface approximation from point cloud data.
 *
 *              Classes that implement this interface can be used for Marching
 *              Cubes based mesh generation algorithms in this toolkit via
 *              the @ref SurfaceReconstruction interface.
 */
template<typename BaseVecT>
class PointsetSurface
{
public:

    // /// Shared pointer type declaration
    // typedef std::shared_ptr<PointsetSurface<BaseVecT>> Ptr;

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
        distance(Vector<BaseVecT> v) const = 0;
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
    virtual Normal<BaseVecT> getInterpolatedNormal(Vector<BaseVecT> position) const;

    /**
     * @brief   Returns the internal point buffer. After a call of
     *          @ref calculateSurfaceNormals the buffer will contain
     *          normal information.
     */
    virtual PointBufferPtr<BaseVecT> pointBuffer() const;

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
    PointsetSurface(PointBufferPtr<BaseVecT> pointcloud);

    PointsetSurface() {};

    /// The point cloud used for surface approximation
    PointBufferPtr<BaseVecT> m_pointBuffer;

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
