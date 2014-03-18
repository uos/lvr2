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


namespace registration
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

    string getDataName() const
    {
        return m_variables["dataCloud"].as<string>();
    }

    double getEpsilon() const
    {
        m_variables["epsilon"].as<double>();
    }

    double getMaxDistance() const
    {
        m_variables["maxDistance"].as<double>();
    }

    double getRx() const
    {
        m_variables["rx"].as<double>();
    }

    double getRy() const
    {
        m_variables["ry"].as<double>();
    }

    double getRz() const
    {
        m_variables["rz"].as<double>();
    }

    double getTx() const
    {
        m_variables["tx"].as<double>();
    }

    double getTy() const
    {
        m_variables["ty"].as<double>();
    }

    double getTz() const
    {
        m_variables["tz"].as<double>();
    }

    int getMaxIterations() const
    {
        return m_variables["maxIterations"].as<int>();
    }

    string getModelName() const
    {
        return m_variables["modelCloud"].as<string>();
    }

private:

	/// The internally used variable map
	variables_map m_variables;

	/// The internally used option description
	options_description m_descr;

	/// The internally used positional option desription
	positional_options_description m_pdescr;

	double      m_epsilon;
	double      m_maxDistance;
	double      m_rx;
	double      m_ry;
	double      m_rz;
	double      m_tx;
	double      m_ty;
	double      m_tz;
	int         m_maxIterations;
	string      m_modelName;
	string      m_dataName;

};

inline ostream& operator<<(ostream& os, const Options& o)
{
    os << "##### Registration Options #####" << endl;
    os << "Epsilon \t\t: " << o.getEpsilon() << endl;
    os << "Max. distance \t\t: " << o.getMaxDistance() << endl;
    os << "Max. iterations \t: " << o.getMaxIterations() << endl;
    os << "Model File \t\t: " << o.getModelName() << endl;
    os << "Data File \t\t: " << o.getDataName() << endl;
    os << "Translation \t\t: " << o.getTx() << " " << o.getTy() << " " << o.getTz() << endl;
    os << "Rotation \t\t: " << o.getRx() << " " << o.getRy() << " " << o.getRz() << endl;
    return os;
}

}

#endif

