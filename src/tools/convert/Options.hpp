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
 *  Created on: Aug 23 2012
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


namespace convert
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

	string  getOutputFile()     const { return m_variables["outputFile"].as<string>();}
	string  getColorMap()       const { return m_variables["intensityColorMap"].as<string>();}
	string  getInputFile()      const { return (m_variables["inputFile"].as< vector<string> >())[0];}

	float   getMaxIntensity()   const { return m_variables["maxIntensity"].as<float>();}
	float   getMinIntensity()   const { return m_variables["minIntensity"].as<float>();}

	bool    convertIntensity()  const { return (m_variables.count("intensityToColor"));}
	bool    printUsage() const;

private:

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
	cout << "##### Input file \t\t: "  << o.getInputFile() << endl;
	cout << "##### Output file \t\t: " 	<< o.getOutputFile() << endl;

	return os;
}

} // namespace reconstruct


#endif /* OPTIONS_H_ */
