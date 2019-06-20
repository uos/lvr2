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
 * PCLFiltering.h
 *
 *  @date 22.11.2011
 *  @author Thomas Wiemann
 */

#ifndef PCLFILTERING_H_
#define PCLFILTERING_H_

#ifdef LVR2_USE_PCL

#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/PointBuffer.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/statistical_outlier_removal.h>

namespace lvr2
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
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr   m_kdTree;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr     m_FilteredCloud;

    bool m_useColors;

};

} /* namespace lvr2 */

#endif

#endif /* PCLFILTERING_H_ */
