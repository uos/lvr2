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
 * PCLFiltering.h
 *
 *  @date 22.11.2011
 *  @author Thomas Wiemann
 */

#ifndef PCLFILTERING_H_
#define PCLFILTERING_H_

#ifdef _USE_PCL_

#include "io/Timestamp.hpp"
#include "io/PointBuffer.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/statistical_outlier_removal.h>

namespace lvr
{

class PCLFiltering
{
public:
    PCLFiltering( PointBufferPtr buffer );
    virtual ~PCLFiltering();

    void applyMLSProjection(float searchRadius);
    void applyOutlierRemoval(int meank, float thresh);

    PointBufferPtr getPointBuffer();

private:
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr       m_pointCloud;
    pcl::PointCloud<pcl::Normal>::Ptr            m_pointNormals;
#ifdef _PCL_VERSION_12_
    pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr      m_kdTree;
#else
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr   m_kdTree;
#endif

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr     m_FilteredCloud;

    bool m_useColors;

};

} /* namespace lvr */

#endif

#endif /* PCLFILTERING_H_ */
