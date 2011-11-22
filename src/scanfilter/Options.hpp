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


namespace filter
{

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
	 * @brief	Returns the used MLSDistance
	 */
	float   mlsDistance()  const;

	/**
	 * @brief   Returns kMean value for outlier removal
	 */
	int   sorMeanK() const;

	/**
	 * @brief   Returns max size of neighbourhood for outlier removal
	 */
	float   sorDevThreshold() const;


	/**
	 * @brief   True if outlier removel is turned on
	 */
	bool    removeOutliers() const;

	/**
	 * @brief   Retuns the input file
	 */
	string inputFile() const;

    /**
     * @brief   Retuns the input file
     */
    string outputFile() const;


	/**
	 * @brief   Prints a usage message
	 */
	bool    printUsage() const;

private:

	/// Max distance used for MLS projection
	float m_mlsDistance;

	/// Output file name
	string m_outputFile;

    /// The internally used variable map
    variables_map                   m_variables;

    /// The internally used option description
    options_description             m_descr;

    /// The internally used positional option desription
    positional_options_description  m_pdescr;

};


/// Overlaoeded outpur operator
inline ostream& operator<<(ostream& os, const Options &o)
{
	cout << "##### Program options: " 	<< endl;
	cout << "##### Input file \t\t: "  << o.inputFile() << endl;
	cout << "##### Output file \t\t: " 	<< o.outputFile() << endl;

	if( o.mlsDistance() > 0)
	{
	    cout << "##### MLS Distance \t\t: " << o.mlsDistance() << endl;
	}
	else
	{
	    cout << "##### Using MLS Projection\t: NO" << endl;
	}
	return os;
}

} // namespace reconstruct


#endif /* OPTIONS_H_ */
