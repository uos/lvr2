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
 * Options.h
 *
 *  Created on: July 14, 2013
 *      Author: Henning Deeken {hdeeken@uos.de}
 *              Ann-Katrin Häuser {ahaeuser@uos.de}
 */

#ifndef OPTIONS_H_
#define OPTIONS_H_

#include <iostream> 
#include <string.h>
#include <vector> 
#include <boost/program_options.hpp>
#include <float.h>
#include <omp.h>
#include <fstream>

using namespace std;

namespace fusion{
using namespace boost::program_options;
/**
 * @brief A class to parse the program options for the fusion
 * 		  executable.
 */
class Options {

public:

	/**
	 * @brief 	Ctor. Parses the command parameters given to the main
	 * 		  	function of the program
	 */
	Options(int argc, char** argv);
	
	virtual ~Options();

	/**
	 * @brief	Prints a usage message to stdout.
	 */
	bool	printUsage() const;

	/**
	 * @brief	Returns true if an output filen name was set
	 */
	bool	outputFileNameSet() const;

	/**
	 * @brief	Returns the output file name
	 */
	string 	getMesh1FileName() const;
	
	/**
	 * @brief	Returns the output file name
	 */
	string 	getMesh2FileName() const;
	
	/**
	 * @brief	Returns the output file name
	 */
	string 	getFusionMeshFileName() const;
	
	
	/**
	 * @brief	Returns the distance treshold file name
	 */
	double getDistanceTreshold() const;

private:

	/// The internally used variable map
	variables_map			        m_variables;

	/// The internally used option description
	options_description 		    m_descr;

	/// The internally used positional option desription
	positional_options_description 	m_pdescr;

};

/// Overlaoeded outpur operator
inline ostream& operator<<(ostream& os, const Options &o)
{
	cout << "##### Program options: " << endl;
	cout << endl;
	cout << "##### Mesh1: " << o.getMesh1FileName() << endl;
	cout << "##### Mesh2: " << o.getMesh2FileName() << endl;
		
	if(o.outputFileNameSet())
	{
	    cout << "##### FusionMesh: " << o.getFusionMeshFileName() << endl;
	}	

/*
	if(o.getIntersections() > 0)
	{
	    cout << "##### Intersections \t\t: " << o.getIntersections() << endl;
	}
	else
	{
	    cout << "##### Voxelsize \t\t: " << o.getVoxelsize() << endl;
	}
*/
	
	return os;
}

} // namespace fusion


#endif /* OPTIONS_H_ */
