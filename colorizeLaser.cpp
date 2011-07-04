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
		for ( int j = 0; j < dummy_count; j++ ) {
			fscanf( f, "%f", &dummy );
			printf( "read dummy...\n" );
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
	i--;
	
	/* Resize cloud to amount of points. */
	cloud->width = i;
	cloud->points.resize( cloud->width * cloud->height );

	printf( "%u values read.\nPointcloud loaded.\n", i );

	if ( f ) {
		fclose( f );
	}



}


int main( int argc, char ** argv ) {

	/* Check, if we got enough command line arguments */
	if ( argc < 3 ) {
		printf( "Usage: %s laserdata kinectdata1 [ kinectdata2 ... ]\n", *argv );
		exit( EXIT_SUCCESS );
	}
	srand (time (NULL));

	PointCloud<PointXYZRGB>::Ptr lasercloud(  new PointCloud<PointXYZRGB> );
	PointCloud<PointXYZRGB>::Ptr kinectcloud( new PointCloud<PointXYZRGB> );

	/* Read clouds from file. */
	printf( "Loading laserscan data...\n" );
	readPts( argv[1], lasercloud );
	printf( "Loading kinect data...\n" );
	for ( int i = 2; i < argc; i++ ) {
		readPts( argv[i], kinectcloud );
	}

	/* Generate octree for kinect pointcloud */
	OctreePointCloud<PointXYZRGB> octree( 128.0f );
	octree.setInputCloud( kinectcloud );
	octree.addPointsFromInputCloud();

	/* Run through laserscan cloud and find neighbours. */

	for ( int i = 0; i < lasercloud->points.size(); i++ ) {

		// K nearest neighbor search

		std::vector<int>   pointIdx;
		std::vector<float> pointSqrDist;

		printf( "Searching neighbours of (%f, %f, %f)\n",
				lasercloud->points[i].x, lasercloud->points[i].y,
				lasercloud->points[i].z );
		if ( octree.nearestKSearch( lasercloud->points[i], 3, pointIdx, pointSqrDist ) > 0 ) {
			printf( "Found %u\n", (unsigned int) pointIdx.size() );
			/* Use color values of all aquired points according to their suared
			 * distance. */
			float dist_sum = 0;
			for ( int k = 0; k < pointIdx.size(); k++ ) {
				dist_sum += pointSqrDist[k];
			}
			float r = 0, g = 0, b = 0;
			for ( int k = 0; k < pointIdx.size(); k++ ) {
				uint32_t rgb = *reinterpret_cast<int*>( &kinectcloud->points[ pointIdx[k] ].rgb );
				printf( "r = %u * %f = %f\n", ( ( rgb >> 16 ) & 0x0000ff ), pointSqrDist[k] / dist_sum, 
						( ( rgb >> 16 ) & 0x0000ff ) * pointSqrDist[k] / dist_sum );
				r += ( ( rgb >> 16 ) & 0x0000ff ) * pointSqrDist[k] / dist_sum;
				g += ( ( rgb >> 8 )  & 0x0000ff ) * pointSqrDist[k] / dist_sum;
				b += ( ( rgb )       & 0x0000ff ) * pointSqrDist[k] / dist_sum;
			}
			/* Now set the color to the laserscan cloud. */
			uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
			lasercloud->points[i].rgb = *reinterpret_cast<float*>( &rgb );
			printf( "% 11f % 11f % 11f %3d %3d %3d\n",
					lasercloud->points[i].x, lasercloud->points[i].y,
					lasercloud->points[i].z, (uint8_t) r, (uint8_t) g, (uint8_t) b );

		}
	}


}
