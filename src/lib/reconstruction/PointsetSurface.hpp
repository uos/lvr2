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
 * SurfaceApproximation.hpp
 *
 *  @date 24.01.2012
 *  @author Thomas Wiemann
 */

#ifndef POINTSETSURFACE_HPP_
#define POINTSETSURFACE_HPP_

#include "io/PointBuffer.hpp"


namespace lssr
{

/**
 * @brief       An interface class to wrap all functionality that is needed
 *              to generate a surface approximation from point cloud data.
 *
 *              Classes that implement this interface can be used for Marching
 *              Cubes based mesh generation algorithms in this toolkit via
 *              the @ref SurfaceReconstruction interface.
 */
template<typename VertexT>
class PointsetSurface
{
public:
    /**
     * @brief   Returns the distance of vertex v from the nearest tangent plane
     * @param   v                     A grid point
     * @param   projectedDistance     Projected distance of the query point to the
     *                              isosurface
     * @param   euclideanDistance     Euklidean Distance to the nearest data point
     */
    virtual void distance(VertexT v,
            float &projectedDistance,
            float &euklideanDistance) = 0;

    /**
     * @brief   Calculates surface normals for each data point in the given
     *          PointBuffeer. If the buffer alreay contains normal information
     *          it will be overwritten with the new normals.
     */
    virtual void calculateSurfaceNormals() = 0;


    /**
     * @brief   Returns the internal point buffer. After a call of
     *          @ref calculateSurfaceNormals the buffer will contain
     *          normal information.
     */

    /**
     * @brief   Retruns the used point buffer
     */
    virtual PointBufferPtr  pointBuffer() { return m_pointBuffer;}

protected:

    /**
     * @brief   Constructor. Stores the given buffer pointer. If the point
     *          buffer does not contain surface normals, you will have to call
     *          @ref calculateSurfaceNormals before the first call @distance.
     */
    PointsetSurface(PointBufferPtr pointcloud)
        : m_pointBuffer(pointcloud) {}

    /// The point cloud used for surface approximation
    PointBufferPtr         m_pointBuffer;
};

} // namespace lssr

#endif /* POINTSETSURFACE_HPP_ */
