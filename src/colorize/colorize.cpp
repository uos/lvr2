/*******************************************************************************
 * Copyright © 2011 Universität Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 59 Temple
 * Place - Suite 330, Boston, MA  02111-1307, USA
 ******************************************************************************/


 /**
 * @file       colorize.cpp
 * @brief      Transfer color information from one to another pointcloud.
 * @details    Takes the color information from one colored point clouds and
 *             transfers these color informations to near points in an
 *             uncolored second point cloud.
 * @author     Lars Kiesow (lkiesow), lkiesow@uos.de
 * @version    111121
 * @date       Created:       2011-07-02 12:18:03
 * @date       Last modified: 2011-11-21 23:29:45
 */

#include <stdio.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>
#include <io/ModelFactory.hpp>
#include <geometry/ColorVertex.hpp>

#include "reconstruction/StannPointCloudManager.hpp"
#include "reconstruction/FastReconstruction.hpp"
#include "io/PLYIO.hpp"

// Optional PCL bindings
#ifdef _USE_PCL_
#include "reconstruction/PCLPointCloudManager.hpp"
#endif


typedef lssr::PointCloudManager<lssr::ColorVertex<float, unsigned char>,
        lssr::Normal<float> >::Ptr PointCloudManagerPtr;


float maxdist = std::numeric_limits<float>::max();
unsigned char rgb[3] = { 255, 0, 0 };
std::string pcm_name = "stann";


/**
 * @brief Prints usage information.
 * @param name  Program name (pass argv[0] to this)
 **/
void printHelp( char * name ) {

	printf( "Usage: %s [options] infile1 infile2 outfile\n"
			"Options:\n"
			"   -h   Show this help and exit.\n"
			"   -d   Maximum distance for neighbourhood.\n"
			"   -p   Set point cloud manager (default: stann).\n"
			"   -j   Number of jobs to be scheduled parallel.\n"
			"        Positive integer or “auto” (default)\n"
			"   -c   Set color of points with no neighbours \n"
			"        as 24 bit hexadecimal integer.\n", name );

}


/**
 * @brief Parse command line arguments.
 **/
void parseArgs( int argc, char ** argv ) {

	/* Parse options */
	char c;
	while ( ( c = getopt( argc, argv, "hd:j:c:p:" ) ) != -1 ) {
		switch (c) {
			case 'h':
				printHelp( *argv );
				exit( EXIT_SUCCESS );
			case 'd':
				maxdist = atof( optarg ) * atof( optarg );
				break;
			case 'p':
				if ( strcmp( optarg, "pcl" ) && strcmp( optarg, "stann" ) ) {
					fprintf( stderr, "Invaild option »%s« for point cloud "
							"manager. Ignoring option.\n", optarg );
					break;
				}
				pcm_name = optarg;
				break;
			case 'm':
				if ( !strcmp( optarg, "auto" ) ) {
					omp_set_num_threads( omp_get_num_procs() );
				} else {
					omp_set_num_threads( 
							atoi( optarg ) > 1 
							? atoi( optarg )
							: omp_get_num_procs() );
				}
				break;
			case 'c':
				uint32_t new_rgb = 0;
				sscanf( optarg, "%x", &new_rgb );
                rgb[0] = (int) ((uint8_t *) &new_rgb)[2];
                rgb[1] = (int) ((uint8_t *) &new_rgb)[1];
                rgb[2] = (int) ((uint8_t *) &new_rgb)[0];
		}
	}

	/* Check, if we got enough command line arguments */
	if ( argc - optind < 3 ) {
		printHelp( *argv );
		exit( EXIT_SUCCESS );
	}
	
}


/**
 * @brief Load a point cloud from a file.
 * @param pc 
 **/
void loadPointCloud( lssr::PointBufferPtr &pc, PointCloudManagerPtr &pcm, char* filename )
{
    
	/* Read clouds from file. */
	printf( "Loading point cloud %s…\n", filename );
    lssr::ModelFactory io_factory;
    lssr::ModelPtr model = io_factory.readModel( filename );
    if ( model && model->m_pointCloud ) 
	{
		pc = model->m_pointCloud;
	}
	else
    {
        printf( "error: Clould not load pointcloud from »%s«", filename );
        exit( EXIT_FAILURE );
    }

    if ( pcm_name == "stann" )
    {
        printf( "Creating STANN point cloud manager…\n" );
        pcm = PointCloudManagerPtr( new lssr::StannPointCloudManager<
            lssr::ColorVertex<float, unsigned char>, 
            lssr::Normal<float> >( pc ) );
    }
#ifdef _USE_PCL_
    else if ( pcm_name == "pcl" ) 
    {
        printf( "Creating STANN point cloud manager…\n" );
        pcm = PointCloudManagerPtr( new lssr::PCLPointCloudManager<
            lssr::ColorVertex<float, unsigned char>, 
            lssr::Normal<float> >( pc ) );
    }
#endif
    else
    {
        printf( "error: Invalid point cloud manager specified.\n" );
        exit( EXIT_FAILURE );
    }

    pcm->setKD( 10 );
    pcm->setKI( 10 );
    pcm->setKN( 10 );

	printf( "Point cloud with %u points loaded…\n", pcm->getNumPoints() );

}


/*******************************************************************************
 *         Name:  main
 *  Description:  Main function.
 ******************************************************************************/
int main( int argc, char ** argv )
{

	omp_set_num_threads( omp_get_num_procs() );
	parseArgs( argc, argv );

	/* Read clouds from file. */
    PointCloudManagerPtr pcm1, pcm2;
	lssr::PointBufferPtr pc1, pc2;
    loadPointCloud( pc1, pcm1, argv[ optind     ] );
    loadPointCloud( pc2, pcm2, argv[ optind + 1 ] );

    /* Colorize first point cloud. */
	printf( "Transfering color information…\n" );
    pcm1->colorizePointCloud( pcm2, maxdist, rgb );

	printf( "Saving new point cloud to »%s«…\n", argv[ optind + 2 ] );
    /* Reset color array of first point cloud. */
    pc1->setIndexedPointColorArray( pcm1->m_colors, pcm1->getNumPoints() );

    /* Save point cloud. */
    lssr::ModelFactory io_factory;
    lssr::ModelPtr model( new lssr::Model( pc1 ) );
    io_factory.saveModel( model, argv[ optind + 2 ] );

    return EXIT_SUCCESS;

}
