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


namespace cl_normals
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

    string  outputFile() const
    {
        return (m_variables["outputFile"].as<string>());
    }



    string  inputFile() const
	{
        return (m_variables["inputFile"].as< vector<string> >())[0];
	}
	
	bool    useRansac() const
	{
		return (m_variables.count("ransac"));
	}
	
	bool	usePCA() const
	{
		return (m_variables.count("pca"));
	}
	
	float 	flipx() const
	{
		return m_variables["flipx"].as<float>();
	}
	
	float 	flipy() const
	{
		return m_variables["flipy"].as<float>();
	}
	
	float 	flipz() const
	{
		return m_variables["flipz"].as<float>();
	}
	
	int     kn() const
	{
        return m_variables["kn"].as<int>();
	}

	int 	ki() const
	{
		return m_variables["ki"].as<int>();
	}

	int		kd() const
	{
		return m_variables["kd"].as<int>();
	}

	float 	getVoxelsize() const
	{
		return m_variables["voxelsize"].as<float>();
	}

	bool 	useVoxelSize() const
	{
		return m_variables.count("voxelsize");
	}

	bool 	reconstruct() const
	{
		return m_variables.count("reconstruct");
	}	

	bool	exportPointNormals() const
	{
		return m_variables.count("exportPointNormals");
	}

private:

	/// The internally used variable map
	variables_map m_variables;

	/// The internally used option description
	options_description m_descr;

	/// The internally used positional option desription
	positional_options_description m_pdescr;

    float         m_flipx;
    float		  m_flipy;
    float 		  m_flipz;
    int			  m_kn;
	int 		  m_ki;
	int			  m_kd;
	float		  m_voxelsize;
    string        m_outputFile;
};

inline ostream& operator<<(ostream& os, const Options& o)
{
    os << "##### OpenCl normal estimation settings #####" << endl;
    if(o.useRansac()){
		os << "Normal Calculation with RANSAC" << endl;
	}else if(o.usePCA()){
		os << "Normal Calculation with PCA" << endl;
	}else{
		os << "Normal Calculation with PCA" << endl;
	}
	os << "Neighbors for normal estimation: "<< o.kn() << endl;
	os << "Neighbors for normal interpolation: " << o.ki() << endl;
	os << "Neighbors for distance function: " << o.kd() << endl;
    os << "Flippoint x: " << o.flipx() << endl;
    os << "Flippoint y: " << o.flipy() << endl;
    os << "Flippoint z: " << o.flipz() << endl;
    
    return os;
}

} // namespace normals

#endif

