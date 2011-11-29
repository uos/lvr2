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
 * PCLPointCloudManager.h
 *
 *  @date 08.08.2011
 *  @author Thomas Wiemann
 */

#ifndef PCLPOINTCLOUDMANAGER_H_
#define PCLPOINTCLOUDMANAGER_H_

#include "PointCloudManager.hpp"

#include "io/PointBuffer.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

namespace lssr
{

template<typename VertexT, typename NormalT>
class PCLPointCloudManager : public PointCloudManager<VertexT, NormalT>
{
public:

    PCLPointCloudManager(PointBuffer* loader, int kn = 10, int ki = 10, int kd = 10);

    virtual ~PCLPointCloudManager();


    /**
     * @brief Returns the k closest neighbor vertices to a given query point
     *
     * @param v         A query vertex
     * @param k         The (max) number of returned closest points to v
     * @param nb        A vector containing the determined closest points
     */
    virtual void getkClosestVertices(const VertexT &v,
            const size_t &k, vector<VertexT> &nb);

    /**
     * @brief Returns the k closest neighbor normals to a given query point
     *
     * @param n         A query vertex
     * @param k         The (max) number of returned closest points to v
     * @param nb        A vector containing the determined closest normals
     */
    virtual void getkClosestNormals(const VertexT &n,
            const size_t &k, vector<NormalT> &nb);

    virtual void distance(VertexT v, float &projectedDistance, float &euklideanDistance);

    virtual void calcNormals();


    /**
     * @brief Returns the all points within a given radius from a given query point
     *
     * @param v         A query vertex
     * @param r         The (max) radius
     * @param nb        A vector containing the determined points
     */
    virtual void radiusSearch(const VertexT &v,
            double r, vector<VertexT> &resV, vector<NormalT> &resN);

private:
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr    m_kdTree;
    pcl::PointCloud<pcl::PointXYZ>::Ptr     m_pointCloud;
    pcl::PointCloud<pcl::Normal>::Ptr       m_pointNormals;
};

} // namespace lssr

#include "PCLPointCloudManager.tcc"

#endif /* PCLPOINTCLOUDMANAGER_H_ */
