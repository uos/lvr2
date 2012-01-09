#ifdef _USE_PCL_

#include "PCDIO.hpp"
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

void lssr::PCDIO::save( string filename,
        std::multimap< std::string, std::string > options, lssr::ModelPtr m )
{
    if ( m )
    {
        m_model = m;
    }

    save( filename );
}


lssr::ModelPtr lssr::PCDIO::read( string filename )
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZ> );

    if ( pcl::io::loadPCDFile<pcl::PointXYZ>( filename, *cloud ) == -1)
    {
        std::cerr << "Couldn't read file “" << filename << "”." << std::endl;
        lssr::ModelPtr m;
        return m;
    }

    coord3fArr points = coord3fArr( new coord<float>[ cloud->points.size() ] );
    /* model->m_pointCloud->setPointColorArray( pointColors, numPoints ); */
    
    for ( size_t i(0); i < cloud->points.size(); i++ )
    {
        points[i].x = cloud->points[i].x;
        points[i].y = cloud->points[i].y;
        points[i].z = cloud->points[i].z;
    }

    lssr::ModelPtr model( new Model( lssr::PointBufferPtr( new lssr::PointBuffer )));
    model->m_pointCloud->setIndexedPointArray( points, cloud->points.size() );
    return model;

}


void lssr::PCDIO::save( string filename )
{

    size_t pointcount(0), buf(0);

    lssr::coord3fArr points;
    lssr::color3bArr pointColors;

    points      = m_model->m_pointCloud->getIndexedPointArray( pointcount );
    pointColors = m_model->m_pointCloud->getIndexedPointColorArray( buf );

    /* We need the same amount of color information and points. */
    if ( pointcount != buf )
    {
        pointColors.reset();
        std::cerr << "Amount of points and color information is"
            " not equal. Color information won't be written.\n";
    }

    if ( !pointColors )
    {

        pcl::PointCloud<pcl::PointXYZ> cloud;

        /* Fill in the cloud data. */
        cloud.width    = pointcount;
        cloud.height   = 1;
        cloud.is_dense = false;
        cloud.points.resize( cloud.width * cloud.height );

        for ( size_t i(0); i < pointcount; i++ )
        {
            cloud.points[i].x = points[i].x;
            cloud.points[i].y = points[i].y;
            cloud.points[i].z = points[i].z;
        }

        pcl::io::savePCDFileASCII( filename, cloud );

    }
    else if ( !pointColors )
    {

        pcl::PointCloud<pcl::PointXYZRGB> cloud;

        /* Fill in the cloud data. */
        cloud.width    = pointcount;
        cloud.height   = 1;
        cloud.is_dense = false;
        cloud.points.resize( cloud.width * cloud.height );

        for ( size_t i(0); i < pointcount; i++ )
        {
            cloud.points[i].x = points[i].x;
            cloud.points[i].y = points[i].y;
            cloud.points[i].z = points[i].z;

            uint8_t* rgb = (uint8_t*) reinterpret_cast<uint8_t*>( 
                    const_cast<float *>( &(cloud.points[i].rgb) ) );

            rgb[2] = pointColors[i].r;
            rgb[1] = pointColors[i].g;
            rgb[0] = pointColors[i].b;
        }

        pcl::io::savePCDFileASCII( filename, cloud );

    }

}

#endif /* _USE_PCL_ */
