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
 *  Created on: Nov 21, 2010
 *      Author: Thomas Wiemann
 */

#ifndef OPTIONS_H_
#define OPTIONS_H_

#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>

using std::ostream;
using std::cout;
using std::endl;
using std::string;
using std::vector;


namespace meshopt{

using namespace boost::program_options;

/**
 * @brief A class to parse the program options for the reconstruction
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
	 * @brief	Returns the number of used threads
	 */
	int 	getNumThreads() const;

	/**
	 * @brief	Prints a usage message to stdout.
	 */
	bool	printUsage() const;

	/**
	 *@brief    Returns true of region coloring is enabled.
	 */
	bool    colorRegions() const;

	/**
	 * @brief 	Returns true if cluster optimization is enabled
	 */
	bool 	optimizePlanes() const;

	/**
	  * @brief Return whether the mesh should be retesselated or not.
	  */
	bool	  retesselate() const;

	/**
	 * @brief  True if region clustering without plane optimization is required.
	 */
	bool 	clusterPlanes() const;

	/**
	 * @brief	Returns the output file name
	 */
	string 	getInputFileName() const;

	/**
	 * @brief	Returns the name of the classifier used to color the mesh
	 */
	string 	getClassifier() const;

	/**
	 * @brief   Returns to number plane optimization iterations
	 */
	int getPlaneIterations() const;

	/**
	 * @brief   Returns the normal threshold for plane optimization.
	 */
	float getNormalThreshold() const;

	/**
	 * @brief   Returns the threshold for the size of small
	 *          region deletion after plane optimization.
	 */
	int   getSmallRegionThreshold() const;

	/**
	 * @brief   Minimum value for plane optimzation
	 */
	int   getMinPlaneSize() const;

	/**
	 * @brief	Number of iterations for contour cleanup
	 */
	int   getCleanContourIterations() const;

	/**
	 * @brief   Returns the number of dangling artifacts to remove from
	 *          a created mesh.
	 */
	int   getDanglingArtifacts() const;

	/**
	 * @brief   Returns the region threshold for hole filling
	 */
	int   getFillHoles() const;

	/**
	 * @brief 	Returns the maximum recursion depth for region growing
	 */
	int getDepth() const;

    /**
     * @brief   Returns the fusion threshold for tesselation
     */
    float getLineFusionThreshold() const;


private:

	/// The number of uesed threads
	int				                m_numThreads;

	/// The internally used variable map
	variables_map			        m_variables;

	/// The internally used option description
	options_description 		    m_descr;

	/// The internally used positional option desription
	positional_options_description 	m_pdescr;

	/// The putput file name for face normals
	string 				            m_faceNormalFile;

	/// The number of used default values
	int                             m_numberOfDefaults;

	/// Whether or not the mesh should be retesselated while being finalized
	bool						     m_retesselate;
	
	/// Number of iterations for plane optimzation
	int                             m_planeIterations;

	/// Threshold for plane optimization
	float                           m_planeNormalThreshold;

	/// Threshold for small ragions
	int                             m_smallRegionThreshold;

	/// Number of dangling artifacts to remove
	int                             m_rda;

	/// Threshold for hole filling
	int                             m_fillHoles;

	/// Threshold for plane optimization
	int                             m_minPlaneSize;

	/// Maximum recursion depth for region growing
	int								m_depth;

	int								m_cleanContourIterations;

	/// Threshold for line fusing when tesselating
	float                           m_lineFusionThreshold;

	/// Name of the classifier object to color the mesh
	string							m_classifier;

};


/// Overlaoeded outpur operator
inline ostream& operator<<(ostream& os, const Options &o)
{
	cout << "##### Program options: " << endl;

	cout << "##### Number of threads \t: "    << o.getNumThreads()      << endl;

	if(o.retesselate())
	{
		cout << "##### Retesselate \t\t: YES"     << endl;
		cout << "##### Line fusion threshold \t: " << o.getLineFusionThreshold() << endl;
	}

	if(o.getFillHoles())
	{
	    cout << "##### Fill holes \t\t: " << o.getFillHoles() << endl;
	}
	else
	{
	    cout << "##### Fill holes \t\t: NO" << endl;
	}

	if(o.getDanglingArtifacts())
	{
	    cout << "##### Remove DAs \t\t: " << o.getDanglingArtifacts() << endl;
	}
	else
	{
	    cout << "##### Remove DAs \t\t: NO" << endl;
	}

	if(o.optimizePlanes())
	{
		cout << "##### Optimize Planes \t\t: YES" << endl;
		cout << "##### Plane iterations\t\t: " << o.getPlaneIterations() << endl;
		cout << "##### Normal threshold \t\t: " << o.getNormalThreshold() << endl;
		cout << "##### Region threshold\t\t: " << o.getSmallRegionThreshold() << endl;
	}

	if(o.getDepth())
	{
	    cout << "##### Recursion depth \t\t: " << o.getDepth() << endl;
	}

	return os;
}

} // namespace reconstruct


#endif /* OPTIONS_H_ */
