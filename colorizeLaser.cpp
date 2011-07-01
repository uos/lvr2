/*******************************************************************************
 *
 *       Filename:  colorizeLaser.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
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


void readPts( char * filename, PointCloud<PointXYZ>::Ptr cloud ) {

	/* Open input file. */
	FILE * f = fopen( filename, "r" );
	if ( !f ) {
		fprintf( stderr, "error: Could not open »%s«.\n", filename );
		exit( EXIT_FAILURE );
	}

	/* Resize cloud. */
	cloud->width = 100000;
	cloud->height = 1;
	cloud->points.resize( cloud->width * cloud->height );

	/* Read values */
	int i = 0;
	while ( !feof( f ) ) {
		fscanf( f, "%f %f %f", &cloud->points[i].x, &cloud->points[i].y,
				&cloud->points[i].z );
		i++;
		/* We jave more points: enlarge cloud */
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

	PointCloud<PointXYZ>::Ptr cloud (new PointCloud<PointXYZ>);

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

	OctreePointCloud<PointXYZ> octree (resolution);

	octree.setInputCloud (cloud);
	octree.addPointsFromInputCloud ();

	PointXYZ searchPoint;

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
