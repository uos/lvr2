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

#include "lvr2/io/PCDIO.hpp"
#include <fstream>
#ifdef LVR2_USE_PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#endif /* LVR2_USE_PCL */

#define isnan(x) ((x) != (x))

namespace lvr2
{

#ifdef LVR2_USE_PCL

ModelPtr PCDIO::read( string filename )
{
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZRGBNormal> );

    if ( pcl::io::loadPCDFile<pcl::PointXYZRGBNormal>( filename, *cloud ) == -1)
    {
        std::cerr << "Couldn't read file “" << filename << "”." << std::endl;
        ModelPtr m;
        return m;
    }

    bool has_normals = false;
    bool has_colors = false;

    floatArr points = floatArr( new float[ cloud->points.size() * 3 ] );
    ucharArr colors = ucharArr( new unsigned char[ cloud->points.size() * 3] );
    floatArr normals =  floatArr( new float[ cloud->points.size() * 3] );
    /* model->m_pointCloud->setPointColorArray( pointColors, numPoints ); */
    for ( size_t i(0); i < cloud->points.size(); i++ )
    {
        if(!isnan(cloud->points[i].x) && !isnan(cloud->points[i].y) && !isnan(cloud->points[i].z)  )
        {
            points[i*3 + 0] = cloud->points[i].x;
            points[i*3 + 1] = cloud->points[i].y;
            points[i*3 + 2] = cloud->points[i].z;
        }
        else
        {
            points[i*3 + 0] = 0.0;
            points[i*3 + 1] = 0.0;
            points[i*3 + 2] = 0.0;
        }

        if(!isnan(cloud->points[i].r) && !isnan(cloud->points[i].g) && !isnan(cloud->points[i].b)  )
        {
            colors[i*3 + 0] = cloud->points[i].r;
            colors[i*3 + 1] = cloud->points[i].g;
            colors[i*3 + 2] = cloud->points[i].b;
            has_colors = true;
        }
        else
        {
            colors[i*3 + 0] = 0;
            colors[i*3 + 1] = 255;
            colors[i*3 + 2] = 0;
        }

        if(!isnan(cloud->points[i].normal_x) && !isnan(cloud->points[i].normal_y) && !isnan(cloud->points[i].normal_z) )
        {
            normals[i*3 + 0] = cloud->points[i].normal_x;
            normals[i*3 + 1] = cloud->points[i].normal_y;
            normals[i*3 + 2] = cloud->points[i].normal_z;
            has_normals = true;
        }
    }

    ModelPtr model( new Model( PointBufferPtr( new PointBuffer )));
    model->m_pointCloud->setPointArray( points, cloud->points.size() );

    if(has_colors)
    {
        model->m_pointCloud->setColorArray( colors, cloud->points.size() );
    }

    if(has_normals)
    {
        model->m_pointCloud->setNormalArray( normals, cloud->points.size() );
    }
    m_model = model;
    return model;

}
#else /*  LVR2_USE_PCL */

ModelPtr PCDIO::read( string filename )
{
    /* Without PCL we do not read pcd files. */
    ModelPtr m( new Model );
    return m;
}
#endif /* LVR2_USE_PCL */


void PCDIO::save( string filename )
{

    size_t pointcount(0), buf(0);
    size_t w_color(0);

    floatArr points;
    ucharArr pointColors;

    pointcount  = m_model->m_pointCloud->numPoints();
    points      = m_model->m_pointCloud->getPointArray();
    pointColors = m_model->m_pointCloud->getUCharArray("colors", buf, w_color);

    /* We need the same amount of color information and points. */
    if ( pointcount != buf )
    {
        pointColors.reset();
        std::cerr << "Amount of points and color information is"
            " not equal. Color information won't be written." << std::endl;
    }

    std::ofstream out( filename.c_str() );

    if ( !out.is_open() )
    {
        std::cerr << "Could not open file »" << filename << "«…"
            << std::endl;
        return;
    }

    out << "# .PCD v.7 - Point Cloud Data file format" << std::endl;
    out << "FIELDS x y z" << ( pointColors ? " rgb" : "" ) << std::endl;
    out << "SIZE 4 4 4" << ( pointColors ? " 4" : "" ) << std::endl;
    out << "TYPE F F F" << ( pointColors ? " F" : "" ) << std::endl;
    out << "WIDTH " << pointcount << std::endl;
    out << "HEIGHT 1" << std::endl;
    out << "POINTS " << pointcount << std::endl;
    out << "DATA ascii" << std::endl;


    for ( size_t i(0); i < pointcount; i++ )
    {
        /* Write coordinates. */
        out << points[i*3 + 0] << " " << points[i*3 + 1] << " " << points[i*3 + 2];

        /* Write color information if there are any. */
        if ( pointColors )
        {
            /* Convert uchar array to float. */
            float rgbf(0);
            uint8_t* rgb = (uint8_t*) reinterpret_cast<uint8_t*>( &rgbf );
            rgb[2] = pointColors[i*w_color + 0];
            rgb[1] = pointColors[i*w_color + 1];
            rgb[0] = pointColors[i*w_color + 2];

            /* Write data. */
            out << " " << rgbf;

        }
        out << std::endl;

    }

    out.close();

}

} // namespace lvr2
