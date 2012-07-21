/*******************************************************************************
 * Copyright © 2012 Universität Osnabrück
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
 * @file       statstrain.cpp
 * @brief      Program to determine coefficients for statistical texture matching
 * @details    
 * @author     Kim Oliver Rinnewitz (krinnewitz), krinnewitz@uos.de
 * @version    120108
 * @date       Created:       2012-07-21 02:49:26
 * @date       Last modified: 2012-07-21 02:49:30
 */

#include <iostream>
#include <io/Timestamp.hpp>
#include <io/TextureIO.hpp>
#include <cstdlib>
#include <iomanip>
#include <sstream>

using namespace std;

/**
 * \brief Main entry point of the program.
**/
int main( int argc, char ** argv )
{

	if (argc != 2)
	{
		cout<<"Usage: "<<argv[0]<<" <filename>"<<endl;
		return EXIT_FAILURE;
	}

	cout<<"Welcome to statstrain - matching textures with a passion!"<<endl;
	cout<<"------------------------------------------------"<<endl;
	lssr::TextureIO* tio = new lssr::TextureIO(argv[1]);

	delete tio;
	return EXIT_SUCCESS;

}
