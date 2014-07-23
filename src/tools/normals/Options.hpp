/**
 * Copyright (C) 2013 Universität Osnabrück
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


#ifndef OPTIONS_H_
#define OPTIONS_H_

#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::ostream;


namespace normals
{

using namespace boost::program_options;

/**
 * @brief A class to parse the program options for the reconstruction
 * executable.
 */
class Options
{
public:

	/**
	 * @brief Ctor. Parses the command parameters given to the main
	 *     function of the program
	 */
	Options(int argc, char** argv);
	virtual ~Options();

	/**
	 * @return  Returns the directory to convert
	 */
	string  getInputDirectory() const
	{
	    return m_variables["inputDirectory"].as<string>();
	}

	/**
	 * @return  Get output file name
	 */
	string  getOutputFile() const
	{
	    return m_variables["outputFile"].as<string>();
	}

	/**
	 * @return  Get target number of points in the output file, i.e.,
	 *          the number of target points for reduction. A negative
	 *          value indicates no reduction.
	 */
	int     getTargetSize() const
	{
	    return m_variables["targetSize"].as<int>();
	}

	/**
	 * @return  Get start scan number
	 */
	int     getStart() const
	{
	    return m_variables["start"].as<int>();
	}

	/**
	 * @return  Get end scan number
	 */
	int     getEnd() const
	{
	    return m_variables["end"].as<int>();
	}

private:

	/// The internally used variable map
	variables_map m_variables;

	/// The internally used option description
	options_description m_descr;

	/// The internally used positional option desription
	positional_options_description m_pdescr;

	/// First scan to read
	int         m_start;

	/// Last scan to read
	int         m_end;

	/// Input directory
	string      m_inputDirectory;

	/// Output file name
	string      m_outputFile;

	/// Target size
	int         m_targetSize;

};

inline ostream& operator<<(ostream& os, const Options& o)
{
    os << "##### Normal Conversion Options #####" << endl;
    os << "Input Dir\t: "   << o.getInputDirectory() << endl;
    os << "Start \t\t: "    << o.getStart() << endl;
    os << "Output File\t: " << o.getOutputFile() << endl;
    os << "End \t: "        << o.getEnd() << endl;
    os << "Target Size\t: " << o.getTargetSize() << endl;
    return os;
}

} // namespace normals

#endif

