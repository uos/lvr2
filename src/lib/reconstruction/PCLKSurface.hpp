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
 * PCLKSurface.hpp
 *
 *  @date 24.01.2012
 *  @author Thomas Wiemann
 */

#ifndef PCLKSURFACE_HPP_
#define PCLKSURFACE_HPP_

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

#include "PointSetSurface.hpp"

#include "io/PointBuffer.hpp"
#include "io/Model.hpp"

namespace lssr
{

/**
 * @brief   PointsetSurface interpolation based on PCL's internal normal
 *          estimation
 */
template<typename VertexT>
class PCLKSurface : public PointsetSurface
{
public:

    /**
     * @brief   Ctor.
     *
     * @param kn    Number of points used for normal estimation
     * @param kd    Number of points used for distance function evaluation
     */
    PCLKSurface( PointBufferPtr loader, int kn = 10, int kd = 10 );

    /**
     * @brief   Returns the distance of vertex v from the nearest tangent plane
     * @param   v                     A grid point
     * @param   projectedDistance     Projected distance of the query point to the
     *                              isosurface
     * @param   euclideanDistance     Euklidean Distance to the nearest data point
     */
    virtual void distance(VertexT v, float &projectedDistance, float &euklideanDistance);

    /**
     * @brief   Calculates surface normals for each data point in the given
     *          PointBuffeer. If the buffer alreay contains normal information
     *          it will be overwritten with the new normals.
     */
    virtual void calculateSurfaceNormals();

    /**
     * @brief   Returns the internal point buffer. After a call of
     *          @ref calculateSurfaceNormals the buffer will contain
     *          normal information.
     */
    virtual PointBufferPtr  pointBuffer();

    virtual ~PCLKSurface();

private:

    /// The FLANN search tree
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr    m_kdTree;

    /// A PCL point cloud representation of the given buffer
    pcl::PointCloud<pcl::PointXYZ>::Ptr     m_pointCloud;

    /// The estimated point normals
    pcl::PointCloud<pcl::Normal>::Ptr       m_pointNormals;

    /// The number of points used for normal estimation
    int                                     m_kn;

    /// The number of points used for normal interpolation
    int                                     m_ki;

    /// The number of points used for distance function evaluation
    int                                     m_kd;
};

} /* namespace lssr */

#include "PCLKSurface.tcc"

#endif /* PCLKSURFACE_HPP_ */
