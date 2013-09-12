/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */
 
 /*
 * Main.cpp
 *
 *  Created on: August 21, 2013
 *      Author: Henning Deeken {hdeeken@uos.de}
 */

// Program options for this tool
#include "Options.hpp"

// Local includes
#include "io/PLYIO.hpp"
#include "io/Timestamp.hpp"
#include "io/Progress.hpp"
#include "io/Model.hpp"
#include "io/ModelFactory.hpp"

#include "slicer/MeshSlicer.hpp"

#include <iostream>

using namespace lssr;

/**
 * @brief   Main entry point for the LSSR fusion executable
 */
int main(int argc, char** argv)
{
	try
	{
		// Parse command line arguments
		slicer::Options options(argc, argv);

		// Exit if options had to generate a usage message
		// (this means required parameters are missing)

		if ( options.printUsage() )
		{
			return 0;
		}

		::std::cout << options << ::std::endl;

		// Create a mesh loader object
		ModelFactory io_factory;
		ModelPtr model = io_factory.readModel( options.getInputFileName() );
		
		MeshBufferPtr input_mesh;
		MeshBufferPtr output_mesh;

		// Parse loaded data
		if ( !model )
		{
			cout << timestamp << "IO Error: Unable to parse " << options.getInputFileName() << endl;
			exit(-1);
		}
		
	    input_mesh = model->m_mesh;
	
		if(!input_mesh)
		{
		    cout << timestamp << "Given file contains no supported mesh information" << endl;
		    exit(-1);
		}	
		
		cout << timestamp << "Successfully loaded mesh." << endl;

		// Create an empty mesh
		
		MeshSlicer mesh;
		cout << "come here" << endl;
		mesh.setDimension(options.getDimension());
		cout << "here" << endl;
		mesh.setValue(options.getValue());
				cout << "here2" << endl;
		// Load and slice mesh
		mesh.addMesh(input_mesh);
		cout << "here3" << endl;
		vector<float> segments = mesh.computeSlice();
		 		cout << "here4" << endl;
		cout << "Slice Segments:" << endl; 	
		for(int i = 0; i < segments.size(); i+=6)
		{
			cout << "(" << segments.at(i) << ", " << segments.at(i+1) << ", " << segments.at(i+2) << ") to (" << segments.at(i+3) << ", " << segments.at(i+4) << ", " << segments.at(i+5) << ")" << endl;
		}	
		 	
     	cout << endl << timestamp << "Program end." << endl;
	}
	catch(...)
	{
		std::cout << "Unable to parse options. Call 'slicer --help' for more information." << std::endl;
	}
	return 0;
}
