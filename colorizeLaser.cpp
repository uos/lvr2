/*******************************************************************************
 *
 *       Filename:  colorizeLaser.cpp
 *
 *    Description:  
 *
 *        Version:  0.1
 *        Created:  07/02/2011 12:18:03 AM
 *       Compiler:  g++
 *
 *         Author:  Lars Kiesow (lkiesow), lkiesow@uos.de
 *        Company:  Universität Osnabrück
 *
 ******************************************************************************/

#include <pcl/point_cloud.h>
#include <pcl/octree/octree.h>

#include <stdio.h>
#include <iostream>
#include <vector>
#include <ctime>

using namespace pcl;
using namespace pcl::octree;


void readPts( char * filename, PointCloud<PointXYZRGB>::Ptr cloud ) {

	/* Open input file. */
	FILE * f = fopen( filename, "r" );
	if ( !f ) {
		fprintf( stderr, "error: Could not open »%s«.\n", filename );
		exit( EXIT_FAILURE );
	}

	/* Determine amount of values per line */
	char line[1024];
	fgets( line, 1023, f );
	int valcount = 0;
	char * pch = strtok( line, "\t " );
	while ( pch ) {
		if ( strcmp( pch, "" ) ) {
			valcount++;
		}
		pch = strtok( NULL, "\t " );
	}

	/* Do we have color information in the pts file? */
	int read_color = valcount >= 6;
	/* Are there additional columns we dont want to have? */
	int dummy_count = valcount - ( read_color ? 6 : 3 );
	float dummy;

	/* Resize cloud. Keep old points. */
	int i = cloud->width;
	cloud->width += 100000;
	cloud->height = 1;
	cloud->points.resize( cloud->width * cloud->height );

	/* Start from the beginning */
	fseek( f, 0, SEEK_SET );

	/* Read values */
	while ( !feof( f ) ) {
		/* Read coordinates */
		fscanf( f, "%f %f %f", &cloud->points[i].x, &cloud->points[i].y,
				&cloud->points[i].z );

		/* Igbore remission, ... */
		for ( i = 0; i < dummy_count; i++ ) {
			fscanf( f, "%f", &dummy );
		}

		/* Read color information, if available */
		if ( read_color ) {
			uint32_t r, g, b;
			fscanf( f, "%u %u %u", &r, &g, &b );
			uint32_t rgb = r << 16 | g << 8 | b;
			cloud->points[i].rgb = *reinterpret_cast<float*>( &rgb );
		}
		i++;
		/* We have more points: enlarge cloud */
		if ( i >= cloud->points.size() ) {
			printf( "%u values read.\n", i );
			cloud->width = cloud->width + 100000;
			cloud->points.resize( cloud->width * cloud->height );
		}
	}
	
	/* Resize cloud to amount of points. */
	cloud->width = i;
	cloud->points.resize( cloud->width * cloud->height );

	printf( "%u values read.\nPointcloud loaded.\n", i );

	if ( f ) {
		fclose( f );
	}



}


int main (int argc, char ** argv) {

	srand (time (NULL));

	PointCloud<PointXYZRGB>::Ptr cloud (new PointCloud<PointXYZRGB>);
	printf( "Size of new cloud: %lu\n", cloud->points.size() );

	// Generate pointcloud data
	cloud->width = 1000;
	cloud->height = 1;
	cloud->points.resize (cloud->width * cloud->height);

	for (size_t i = 0; i < cloud->points.size (); ++i) {
		cloud->points[i].x = 1024.0f * rand () / (RAND_MAX + 1.0);
		cloud->points[i].y = 1024.0f * rand () / (RAND_MAX + 1.0);
		cloud->points[i].z = 1024.0f * rand () / (RAND_MAX + 1.0);
	}

	float resolution = 128.0f;

	OctreePointCloud<PointXYZRGB> octree( resolution );

	octree.setInputCloud(cloud);
	octree.addPointsFromInputCloud ();

	PointXYZRGB searchPoint;

	searchPoint.x = 1024.0f * rand () / (RAND_MAX + 1.0);
	searchPoint.y = 1024.0f * rand () / (RAND_MAX + 1.0);
	searchPoint.z = 1024.0f * rand () / (RAND_MAX + 1.0);

	// K nearest neighbor search

	std::vector<int>   pointIdx;
	std::vector<float> pointSqrDist;

	std::cerr << "K nearest neighbor search at (" << searchPoint.x 
		<< " " << searchPoint.y 
		<< " " << searchPoint.z
		<< ") with K=" << 3 << std::endl;

	if ( octree.nearestKSearch( searchPoint, 3, pointIdx, pointSqrDist ) > 0 ) {
		for (size_t i = 0; i < pointIdx.size(); ++i) {
			printf( "% 9f % 9f % 9f (distance: % 9f)\n", 
					cloud->points[ pointIdx[i] ].x, cloud->points[ pointIdx[i] ].y,
					cloud->points[ pointIdx[i] ].z, pointSqrDist[i] );
		}
	}

}
