/*******************************************************************************
 *
 *       Filename:  colorizeLaser.cpp
 *
 *    Description:  Takes the color information from one or more colored
 *    (kinect-) pointclouds and transfers these color informations to near
 *    points in an uncolored (laser-) cloud.
 *
 *        Version:  0.2
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
        lssr::Normal<float> >* pcm_p;

typedef lssr::PointBuffer* pc_p;


float maxdist = std::numeric_limits<float>::max();
unsigned char rgb[3] = { 255, 0, 0 };


/*******************************************************************************
 *         Name:  printHelp
 *  Description:  Prints usage information.
 ******************************************************************************/
void printHelp( char * name ) {

	printf( "Usage: %s [options] infile1 infile2 outfile\n"
			"Options:\n"
			"   -h   Show this help and exit.\n"
			"   -d   Maximum distance for neighbourhood.\n"
			"   -j   Number of jobs to be scheduled parallel.\n"
			"        Positive integer or “auto” (default)\n"
			"   -c   Set color of points with no neighbours \n"
			"        as 24 bit hexadecimal integer.\n", name );

}


/*******************************************************************************
 *         Name:  printHelp
 *  Description:  Prints usage information.
 ******************************************************************************/
void parseArgs( int argc, char ** argv ) {

	/* Parse options */
	char c;
	while ( ( c = getopt( argc, argv, "hd:j:c:" ) ) != -1 ) {
		switch (c) {
			case 'h':
				printHelp( *argv );
				exit( EXIT_SUCCESS );
			case 'd':
				maxdist = atof( optarg );
				break;
			case 'm':
				if ( !strcmp( optarg, "auto" ) ) {
					omp_set_num_threads( omp_get_num_procs() );
				} else {
					omp_set_num_threads( 
							atoi( optarg ) > 1 ? atoi( optarg ) 
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


/*******************************************************************************
 *         Name:  loadPointCloud
 *  Description:  
 ******************************************************************************/
void loadPointCloud( pc_p* pc, pcm_p* pcm, char* filename )
{
    
	/* Read clouds from file. */
	printf( "Loading pointcloud %s…\n", filename );
    lssr::ModelFactory io_factory;
    lssr::Model* model = io_factory.readModel( filename );
    *pc = model ? model->m_pointCloud : NULL;
    if ( !(*pc) )
    {
        printf( "error: Clould not load pointcloud from »%s«", filename );
        exit( EXIT_FAILURE );
    }

    std::string pcm_name = "stann";
    if ( pcm_name == "stann" )
    {
        printf( "Creating STANN point cloud manager…\n" );
        *pcm = new lssr::StannPointCloudManager<
            lssr::ColorVertex<float, unsigned char>, 
            lssr::Normal<float> >( *pc );
    }
#ifdef _USE_PCL_
    else if ( pcm_name == "pcl" ) 
    {
        printf( "Creating STANN point cloud manager…\n" );
        *pcm = new lssr::PCLPointCloudManager<
            lssr::ColorVertex<float, unsigned char>, 
            lssr::Normal<float> >( *pc );
    }
#endif
    else
    {
        printf( "error: Invalid point cloud manager specified.\n" );
        exit( EXIT_FAILURE );
    }

    (*pcm)->setKD( 10 );
    (*pcm)->setKI( 10 );
    (*pcm)->setKN( 10 );

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
    pcm_p pcm1 = NULL, pcm2 = NULL;
    pc_p  pc1  = NULL, pc2  = NULL;
    loadPointCloud( &pc1, &pcm1, argv[ optind     ] );
    loadPointCloud( &pc2, &pcm2, argv[ optind + 1 ] );

    /* Colorize first point cloud. */
    pcm1->colorizePointCloud( pcm2, maxdist, rgb );

    /* Reset color array of first point cloud. */
    pc1->setPointColorArray( *(pcm1->m_colors), pcm1->getNumPoints() );

    lssr::ModelFactory io_factory2;
    lssr::Model model2;
	lssr::PointBuffer pb2;
	pb2.setPointArray( *(pcm2->m_points), pcm2->getNumPoints() );
    model2.m_pointCloud = &pb2;
    io_factory2.saveModel( &model2, "123.ply" );

    /* Save point cloud. */
    lssr::ModelFactory io_factory;
    lssr::Model model;
    model.m_pointCloud = pc1;
    io_factory.saveModel( &model, argv[ optind + 2 ] );

    return EXIT_SUCCESS;

}
