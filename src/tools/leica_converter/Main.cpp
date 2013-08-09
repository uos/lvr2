/* Copyright (C) 2013 Uni Osnabrück
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
 *
 * Main.cpp
 *
 *  Created on: Aug 9, 2013
 *      Author: Thomas Wiemann
 */

#include <iostream>
#include <algorithm>
#include <string>
#include <stdio.h>
using namespace std;

#include <boost/filesystem.hpp>

#include "Options.hpp"
#include "io/BaseIO.hpp"
#include "io/DatIO.hpp"
#include "io/Timestamp.hpp"
#include "io/ModelFactory.hpp"
#include "io/AsciiIO.hpp"
#include "reconstruction/PCLFiltering.hpp"

using namespace lvr;

ModelPtr filterModel(ModelPtr p, int k, float sigma)
{
	if(p)
	{
		if(p->m_pointCloud)
		{
			PCLFiltering filter(p->m_pointCloud);
			filter.applyOutlierRemoval(k, sigma);
			PointBufferPtr pb( filter.getPointBuffer() );
			ModelPtr out_model( new Model( pb ) );
			return out_model;
		}
	}
	p;
}

int main(int argc, char** argv)
{
	// Parse command line arguments
	leica_convert::Options options(argc, argv);

	boost::filesystem::path inputDir(options.getInputDir());
	boost::filesystem::path outputDir(options.getOutputDir());

	// Check input directory
	if(!boost::filesystem::exists(inputDir))
	{
		cout << timestamp << "Error: Directory " << options.getInputDir() << " does not exist" << endl;
		exit(-1);
	}

	// Check if output dir exists
	if(!boost::filesystem::exists(outputDir))
	{
		cout << timestamp << "Creating directory " << options.getOutputDir() << endl;
		if(!boost::filesystem::create_directory(outputDir))
		{
			cout << timestamp << "Error: Unable to create " << options.getOutputDir() << endl;
			exit(-1);
		}
	}

	// Create director iterator and parse supported file formats
	boost::filesystem::directory_iterator end;
	vector<boost::filesystem::path> v;
	for(boost::filesystem::directory_iterator it(inputDir); it != end; ++it)
	{
		string extension = "";
		if(options.getInputFormat() == "PLY")
		{
			extension = ".ply";
		}
		else if(options.getInputFormat() == "DAT")
		{
			extension = ".dat";
		}
		else if(options.getInputFormat() == "ALL")
		{

		}

		if(it->path().extension() == extension)
		{
			v.push_back(it->path());
		}
	}

	// Sort entries
	sort(v.begin(), v.end());

	int c = 0;
	for(vector<boost::filesystem::path>::iterator it = v.begin(); it != v.end(); it++)
	{
		if(options.getOutputFormat() == "SLAM")
		{
			int reduction = options.getTargetSize();
			ModelPtr model;

			if(reduction == 0)
			{
				cout << timestamp << "Reading point cloud data from " << it->c_str() << "." << endl;
				model = ModelFactory::readModel(string(it->c_str()));
				if(model)
				{
					char name[1024];
					sprintf(name, "%s/scan%03d.3d", it->c_str(), c);
					cout << name << endl;
				}
			}
			else
			{
				cout << "OK " << reduction << endl;
				if(options.getInputFormat() == "DAT")
				{
					DatIO io;
					cout << timestamp << "Reading point cloud data from " << it->c_str() << "." << endl;
					model = io.read(string(it->c_str()), 4, reduction);

					if(options.filter())
					{
						cout << timestamp << "Filtering input data..." << endl;
						model = filterModel(model, options.getK(), options.getSigma());
					}

					if(model)
					{
						char name[1024];
						sprintf(name, "%s/scan%03d.3d", it->c_str(), c);
						cout << "Saving " << name << "..." << endl;
						AsciiIO outIO;
						outIO.save(name);
					}

				}
				else
				{
					cout << timestamp << "Reduction mode currently only supported for DAT format." << endl;
					exit(-1);
				}
			}
			c++;

		}
	}

	return 0;
}


