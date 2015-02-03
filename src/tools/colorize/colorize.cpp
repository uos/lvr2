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

#include <iostream>
#include <cmath>
#include <io/ModelFactory.hpp>
#include <io/Timestamp.hpp>

#include "config/lvropenmp.hpp"

#include "reconstruction/AdaptiveKSearchSurface.hpp"

using namespace lvr;

typedef AdaptiveKSearchSurface<lvr::ColorVertex<float, unsigned char>,
        Normal<float> >::Ptr PointCloudManagerPtr;


float maxdist = std::numeric_limits<float>::max();
unsigned char rgb[3] = { 255, 0, 0 };
std::string pcm_name = "stann";
std::string ply_mode = "PLY_LITTLE_ENDIAN";


/**
 * @brief Prints usage information.
 * @param name  Program name (pass argv[0] to this)
 **/
void printHelp( char * name ) {

    std::cout << "Usage: " << name << " [options] infile1 infile2 outfile" << std::endl
            << "Options:" << std::endl
            << "   -h   Show this help and exit." << std::endl
            << "   -d   Maximum distance for neighbourhood." << std::endl
            << "   -p   Set point cloud manager (default: stann)." << std::endl
            << "   -m   Set mode of PLY output files. If output file" << std::endl
            << "        format is not PLY this option will have no effect." << std::endl
            << "   -j   Number of jobs to be scheduled parallel." << std::endl
            << "        Positive integer or “auto” (default)" << std::endl
            << "   -c   Set color of points with no neighbours " << std::endl
            << "        as 24 bit hexadecimal integer." << std::endl;

}


/**
 * @brief Parse command line arguments.
 **/
void parseArgs( int argc, char ** argv ) {

    /* Parse options */
    char c;
    while ( ( c = getopt( argc, argv, "hd:j:c:p:m:" ) ) != -1 ) {
        switch (c) {
            case 'h':
                printHelp( *argv );
                exit( EXIT_SUCCESS );
            case 'd':
                maxdist = atof( optarg ) * atof( optarg );
                break;
            case 'p':
                if ( strcmp( optarg, "pcl" ) && strcmp( optarg, "stann" ) ) {
                    std::cerr << "Invaild option »" << optarg << "« for point cloud "
                            << "manager. Ignoring option." << std::endl;
                    break;
                }
                pcm_name = optarg;
                break;
            case 'm':
                ply_mode = std::string( optarg );
                break;
            case 'j':
                if ( !strcmp( optarg, "auto" ) ) {
                	OpenMPConfig::setMaxNumThreads();
                } else {
                    OpenMPConfig::setNumThreads(
                            atoi( optarg ) > 1 
                            ? atoi( optarg )
                            : OpenMPConfig::getNumThreads() );
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
void loadPointCloud( lvr::PointBufferPtr &pc, PointCloudManagerPtr &pcm, char* filename )
{
    
    /* Read clouds from file. */
    std::cout << lvr::timestamp <<  "Loading point cloud »" << filename
        << "«…" << std::endl;
    ModelFactory io_factory;
    ModelPtr model = io_factory.readModel( filename );
    if ( model && model->m_pointCloud ) 
    {
        pc = model->m_pointCloud;
    }
    else
    {
        std::cerr << timestamp << "Clould not load pointcloud from »"
            << filename << "«" << std::endl;
        exit( EXIT_FAILURE );
    }

    pcm = PointCloudManagerPtr( new AdaptiveKSearchSurface<ColorVertex<float, unsigned char>, Normal<float> >( pc, pcm_name ));

    pcm->setKD( 10 );
    pcm->setKI( 10 );
    pcm->setKN( 10 );

    std::cout << timestamp << "Point cloud with " << pcm->getNumPoints()
        << " points loaded…" << std::endl;

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
    PointBufferPtr pc1, pc2;
    loadPointCloud( pc1, pcm1, argv[ optind     ] );
    loadPointCloud( pc2, pcm2, argv[ optind + 1 ] );

    /* Colorize first point cloud. */
    std::cout << lvr::timestamp << "Transfering color information…"
        << std::endl;
    pcm1->colorizePointCloud( pcm2, maxdist, rgb );

    std::cout << lvr::timestamp << "Saving new point cloud to »"
        << argv[ optind + 2 ] << "«…" << std::endl;
    /* Reset color array of first point cloud. */
    pc1->setIndexedPointColorArray( pcm1->m_colors, pcm1->getNumPoints() );

    /* Save point cloud. */
    ModelFactory io_factory;
    ModelPtr model( new lvr::Model( pc1 ) );

    
    std::multimap< std::string, std::string > save_opts;
    /* Build call string */
    {
        std::string s("");
        for ( size_t i(0); i < argc-1; i++ )
        {
            s += std::string( argv[i] ) + " ";
        }
        s += argv[ argc-1 ];
        save_opts.insert( pair< std::string, std::string >( "comment", s ) );
    }
    save_opts.insert( pair< std::string, std::string >( "comment",
                "Created with las-vegas-reconstruction (colorize): "
                "http://las-vegas.uos.de/" ) );
    save_opts.insert( pair<std::string, std::string>( "ply_mode", ply_mode ));

    io_factory.saveModel( model, argv[ optind + 2 ] );

    return EXIT_SUCCESS;

}
