/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

#include "lvr2/config/BaseOption.hpp"

using std::ostream;
using std::cout;
using std::endl;
using std::string;
using std::vector;


namespace kaboom
{

using namespace boost::program_options;

/**
 * @brief A class to parse the program options for the reconstruction
 * 		  executable.
 */
class Options : public lvr2::BaseOption
{
public:

	/**
	 * @brief 	Ctor. Parses the command parameters given to the main
	 * 		  	function of the program
	 */
	Options(int argc, char** argv);
	virtual ~Options();

	string	getInputDir() const;
	string 	getOutputDir() const;
	string 	getOutputFormat() const;
    string 	getInputFile() const;
    string 	getOutputFile() const;

    bool    exportScanPositions() const;
	bool	filter() const;
    bool    transformBefore() const;
	int		getK() const;
	float	getSigma() const;
	int		getTargetSize() const;
	double  getVoxelSize() const;
	size_t  getMinPointsPerVoxel() const;

	bool    convertToLVR() const;

	/**
	 * @brief   Returns the position of the x coordinate in the data.
	 */
	int r() const { return m_variables["rPos"].as<int>() ;}

	/**
	 * @brief   Returns the position of the x coordinate in the data.
	 */
	int g() const { return m_variables["gPos"].as<int>();}

	/**
	 * @brief   Returns the position of the x coordinate in the data.
	 */
	int b() const { return m_variables["bPos"].as<int>();}

	/**
	 * @brief   Returns the position of the x coordinate in the data.
	 */
	int i() const { return m_variables["iPos"].as<int>();}
    
    int getStart() const { return m_variables["start"].as<int>(); }

    int getEnd() const { return m_variables["end"].as<int>(); }

	std::string getScanPrefix() const { return m_variables["scanPrefix"].as<std::string>(); }
	std::string getPosePrefix() const { return m_variables["posePrefix"].as<std::string>(); }
	std::string getScanExtension() const { return m_variables["scanExtension"].as<std::string>(); }
	std::string getPoseExtension() const { return m_variables["poseExtension"].as<std::string>(); }

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
	cout << "##### Input dir \t\t: "  << o.getInputDir() << endl;
	cout << "##### Output dir \t\t: " 	<< o.getOutputDir() << endl;
	if(o.filter())
	{
		cout << "##### Filter  \t\t\t: YES" << endl;
		cout << "##### K  \t\t\t\t: " << o.getK() << endl;
		cout << "##### Sigma \t\t\t: " << o.getSigma() << endl;
	}
	else
	{
		cout << "##### Filter  \t\t\t: NO" << endl;
	}
	cout << "##### Target Size \t: " << o.getTargetSize() << endl;
	return os;
}

} // namespace reconstruct


#endif /* OPTIONS_H_ */
