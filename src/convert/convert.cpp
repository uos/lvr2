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
 * @file       convert.cpp
 * @brief      Converts meshes and point clouds from one file format to another.
 * @details    
 * @author     Lars Kiesow (lkiesow), lkiesow@uos.de
 * @version    120108
 * @date       Created:       2012-01-08 02:49:26
 * @date       Last modified: 2012-01-08 02:49:30
 */

#include <cstdio>
#include <io/ModelFactory.hpp>


int main( int argc, char ** argv )
{

    if ( argc != 3 )
    {
        printf( "Usage: %s [options] infile outfile\n", *argv );
        return EXIT_SUCCESS;
    }

    /* Convert point cloud. */
    lssr::ModelFactory io_factory;
    
    printf( "Loading point cloud from „%s“…\n", argv[1] );
    lssr::ModelPtr model( io_factory.readModel( argv[1] ) );

    printf( "Writing point cloud to „%s“…\n", argv[2] );
    io_factory.saveModel( model, argv[2] );

    return EXIT_SUCCESS;

}
