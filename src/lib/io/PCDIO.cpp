
#include "PCDIO.hpp"
#include <fstream>
#ifdef _USE_PCL_
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#endif /* _USE_PCL_ */

void lssr::PCDIO::save( string filename,
        std::multimap< std::string, std::string > options, lssr::ModelPtr m )
{
    if ( m )
    {
        m_model = m;
    }

    save( filename );
}


#ifdef _USE_PCL_
lssr::ModelPtr lssr::PCDIO::read( string filename )
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZRGB> );

    if ( pcl::io::loadPCDFile<pcl::PointXYZRGB>( filename, *cloud ) == -1)
    {
        std::cerr << "Couldn't read file “" << filename << "”." << std::endl;
        lssr::ModelPtr m;
        return m;
    }

    coord3fArr points = coord3fArr( new coord<float>[ cloud->points.size() ] );
    color3bArr colors = color3bArr( new color<uchar>[ cloud->points.size() ] );
    /* model->m_pointCloud->setPointColorArray( pointColors, numPoints ); */
    for ( size_t i(0); i < cloud->points.size(); i++ )
    {
        if(!isnan(cloud->points[i].x) && !isnan(cloud->points[i].y) && !isnan(cloud->points[i].z)  )
        {
            points[i].x = cloud->points[i].x;
            points[i].y = cloud->points[i].y;
            points[i].z = cloud->points[i].z;
            colors[i].r = cloud->points[i].r;
            colors[i].g = cloud->points[i].g;
            colors[i].b = cloud->points[i].b;
        }
        else
        {
            points[i].x = 0.0;
            points[i].y = 0.0;
            points[i].z = 0.0;
            colors[i].r = 0;
            colors[i].g = 255;
            colors[i].b = 0;
        }
    }

    lssr::ModelPtr model( new Model( lssr::PointBufferPtr( new lssr::PointBuffer )));
    model->m_pointCloud->setIndexedPointArray( points, cloud->points.size() );
    model->m_pointCloud->setIndexedPointColorArray( colors, cloud->points.size() );
    m_model = model;
    return model;

}
#else /*  _USE_PCL_ */

lssr::ModelPtr lssr::PCDIO::read( string filename )
{
    /* Without PCL we do not read pcd files. */
    lssr::ModelPtr m( new Model );
    return m;
}
#endif /* _USE_PCL_ */


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
        out << points[i].x << " " << points[i].y << " " << points[i].z;

        /* Write color information if there are any. */
        if ( pointColors )
        {
            /* Convert uchar array to float. */
            float rgbf(0);
            uint8_t* rgb = (uint8_t*) reinterpret_cast<uint8_t*>( &rgbf );
            rgb[2] = pointColors[i].r;
            rgb[1] = pointColors[i].g;
            rgb[0] = pointColors[i].b;

            /* Write data. */
            out << " " << rgbf;

        }
        out << std::endl;

    }

    out.close();

}
