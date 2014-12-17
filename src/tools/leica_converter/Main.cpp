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
#ifdef _USE_PCL_
	#include "reconstruction/PCLFiltering.hpp"
#endif


using namespace lvr;

ModelPtr filterModel(ModelPtr p, int k, float sigma)
{
	if(p)
	{
		if(p->m_pointCloud)
		{
#ifdef _USE_PCL_
			PCLFiltering filter(p->m_pointCloud);
			cout << timestamp << "Filtering outliers with k=" << k << " and sigma=" << sigma << "." << endl;
			size_t original_size = p->m_pointCloud->getNumPoints();
			filter.applyOutlierRemoval(k, sigma);
			PointBufferPtr pb( filter.getPointBuffer() );
			ModelPtr out_model( new Model( pb ) );
			cout << timestamp << "Filtered out " << original_size - out_model->m_pointCloud->getNumPoints() << " points." << endl;
			return out_model;
#else 
			cout << timestamp << "Can't create a PCL Filter without PCL installed." << endl;
			return NULL;
#endif
			
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
		else if(options.getInputFormat() == "TXT")
		{
			extension = ".txt";
		}
		else if(options.getOutputFormat() == "ALL")
		{
			// Filter supported file formats
			if(it->path().extension() == ".ply" || it->path().extension() == ".txt" || it->path().extension() == ".dat")
			{
				extension = string(it->path().extension().c_str());
			}
		}

		if(it->path().extension() == extension)
		{
			v.push_back(it->path());
		}
	}

	// Sort entries
	sort(v.begin(), v.end());

	vector<float>	 		merge_points;
	vector<unsigned char>	merge_colors;

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
					sprintf(name, "%s/scan%03d.3d", outputDir.c_str(), c);
					cout << name << endl;
				}
			}
			else
			{
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
				}
				else
				{
					cout << timestamp << "Reduction mode currently only supported for DAT format." << endl;
					exit(-1);
				}

				if(model)
				{
				/*	// Convert to slam coordinate system
					if(model->m_pointCloud)
					{
						float point[3];
						PointBufferPtr p_ptr = model->m_pointCloud;
						size_t num;
						floatArr points = p_ptr->getPointArray(num);
						for(int i = 0; i < num; i++)
						{
							point[0] = points[3 * i + 1];
							point[1] = points[3 * i + 2];
							point[2] = points[3 * i];

							point[0] *= -100;
							point[1] *= 100;
							point[2] *= 100;

							points[3 * i] = point[0];
							points[3 * i + 1] = point[1];
							points[3 * i + 2] = point[2];
						}
					}
*/
					char name[1024];
					sprintf(name, "%s/scan%03d.3d", outputDir.c_str(), c);
					cout << timestamp << "Saving " << name << "..." << endl;
					AsciiIO outIO;
					outIO.setModel(model);
					outIO.save(name);
				}



			}
			c++;

		}
		else if(options.getOutputFormat() == "MERGE")
		{
			ModelPtr model = ModelFactory::readModel(string(it->c_str()));
			if(model)
			{
				PointBufferPtr points = model->m_pointCloud;
				size_t num_points = 0;
				size_t num_colors = 0;
				floatArr point_arr = points->getPointArray(num_points);
				ucharArr color_arr = points->getPointColorArray(num_colors);

				cout << timestamp << "Adding " << it->c_str() << " to merged point cloud" << endl;

				for(size_t i = 0; i < num_points; i++)
				{
					merge_points.push_back(point_arr[3 * i]);
					merge_points.push_back(point_arr[3 * i + 1]);
					merge_points.push_back(point_arr[3 * i + 2]);

					if(num_points == num_colors)
					{
						merge_colors.push_back(color_arr[3 * i]);
						merge_colors.push_back(color_arr[3 * i + 1]);
						merge_colors.push_back(color_arr[3 * i + 2]);
					}
					else
					{
						for(int j = 0; j < 3; j++)
						{
							merge_colors.push_back(128);
						}
					}
				}
			}
			else
			{
				cout << "Unable to model data from " << it->c_str() << endl;
			}
		}
	}

	if(merge_points.size() > 0)
	{
		cout << timestamp << "Building merged model..." << endl;
		cout << timestamp << "Merged model contains " << merge_points.size() << " points." << endl;

		floatArr points (new float[merge_points.size()]);
		ucharArr colors (new unsigned char[merge_colors.size()]);

		for(size_t i = 0; i < merge_points.size(); i++)
		{
			points[i] = merge_points[i];
			colors[i] = merge_colors[i];
		}

		PointBufferPtr pBuffer(new PointBuffer);
		pBuffer->setPointArray(points, merge_points.size() / 3);
		pBuffer->setPointColorArray(colors, merge_colors.size() / 3);

		ModelPtr model(new Model(pBuffer));

		cout << timestamp << "Writing 'merge.ply'" << endl;
		ModelFactory::saveModel(model, "merge.3d");

	}
	cout << timestamp << "Program end." << endl;
	return 0;
}


