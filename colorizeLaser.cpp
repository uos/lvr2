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
#include <pcl/kdtree/kdtree_flann.h>

#include <stdio.h>
#include <iostream>
#include <vector>
#include <ctime>

using namespace pcl;


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


void printHelp( char * name ) {

	printf( "Usage: %s [options] laserdat kinectdat1 [kinectdat2 ...] outfile\n"
			"Options:\n"
			"   -h   Show this help and exit.\n"
			"   -d   Maximum distance for neighbourhood.\n"
			"   -j   Number of jobs to be scheduled parallel.\n"
			"        Positive integer or “auto” (default)\n"
			"   -c   Set color of points with no neighbours \n"
			"        as 24 bit hexadecimal integer.\n", name );

}




int main( int argc, char ** argv ) {

	double maxdist = std::numeric_limits<double>::max();
	int jobs = omp_get_num_procs();
	uint8_t nc_r = 0, nc_g = 0, nc_b = 0;

	/* Parse options */
	char c;
	while ( ( c = getopt( argc, argv, "hd:j:c:" ) ) != -1 ) {
		switch (c) {
			case 'h':
				printHelp( *argv );
				exit( EXIT_SUCCESS );
			case 'd':
				maxdist = atof( optarg );
				maxdist *= maxdist;
				break;
			case 'm':
				if ( !strcmp( optarg, "auto" ) ) {
					jobs = omp_get_num_procs();
				} else {
					jobs = atoi( optarg ) > 1 
						? atoi( optarg ) 
						: omp_get_num_procs();
				}
				break;
			case 'c':
				uint32_t nc_rgb = 0;
				sscanf( optarg, "%x", &nc_rgb );
				nc_r = ((uint8_t *) &nc_rgb)[2];
				nc_g = ((uint8_t *) &nc_rgb)[1];
				nc_b = ((uint8_t *) &nc_rgb)[0];
		}
	}

	/* Check, if we got enough command line arguments */
	if ( argc - optind < 3 ) {
		printHelp( *argv );
		exit( EXIT_SUCCESS );
	}

	PointCloud<PointXYZRGB>::Ptr lasercloud(  new PointCloud<PointXYZRGB> );
	PointCloud<PointXYZRGB>::Ptr kinectcloud( new PointCloud<PointXYZRGB> );

	/* Read clouds from file. */
	printf( "Loading laserscan data...\n" );
	readPts( argv[optind], lasercloud );
	printf( "Loading kinect data...\n" );
	for ( int i = optind + 1; i < argc - 1; i++ ) {
		readPts( argv[i], kinectcloud );
	}

	/* Generate octree for kinect pointcloud */
	KdTreeFLANN<PointXYZRGB> kdtree; /* param: sorted */
	kdtree.setInputCloud( kinectcloud );

	/* Open output file. */
	FILE * out = fopen( argv[ argc - 1 ], "w" );
	if ( !out ) {
		fprintf( stderr, "error: Could not open »%s«.\n", argv[ argc - 1 ] );
		exit( EXIT_FAILURE );
	}

	/* set number of threads to the number of available processors/cores */
	omp_set_num_threads( jobs );

	printf( "Adding color information...\n" );

	/* Run through laserscan cloud and find neighbours. */
	#pragma omp parallel for
	for ( int i = 0; i < lasercloud->points.size(); i++ ) {

		std::vector<int>   pointIdx(1);
		std::vector<float> pointSqrDist(1);

		/* nearest neighbor search */
		if ( kdtree.nearestKSearch( *lasercloud, i, 1, pointIdx, pointSqrDist ) ) {
			if ( pointSqrDist[0] > maxdist ) {
				fprintf( out, "% 11f % 11f % 11f % 14f % 3d % 3d % 3d\n",
						lasercloud->points[i].x, lasercloud->points[i].y,
						lasercloud->points[i].z, pointSqrDist[0],
						nc_r, nc_g, nc_b );
			} else {
				uint8_t * rgb = (uint8_t *) &kinectcloud->points[ pointIdx[0] ].rgb;
				/* lasercloud->points[i].rgb = kinectcloud->points[ pointIdx[0] ].rgb; */
				fprintf( out, "% 11f % 11f % 11f % 14f % 3d % 3d % 3d\n",
						lasercloud->points[i].x, lasercloud->points[i].y,
						lasercloud->points[i].z, pointSqrDist[0],
						rgb[2], rgb[1], rgb[0] );
			}

		}
	}

	if ( out ) {
		fclose( out );
	}

}
