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
#include <fstream>

using namespace std;

#include <boost/filesystem.hpp>

#include "Options.hpp"
#include <lvr/io/BaseIO.hpp>
#include <lvr/io/DatIO.hpp>
#include <lvr/io/Timestamp.hpp>
#include <lvr/io/ModelFactory.hpp>
#include <lvr/io/AsciiIO.hpp>
#ifdef LVR_USE_PCL
	#include <lvr/reconstruction/PCLFiltering.hpp>
#endif


using namespace lvr;

ModelPtr filterModel(ModelPtr p, int k, float sigma)
{
	if(p)
	{
		if(p->m_pointCloud)
		{
#ifdef LVR_USE_PCL
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
		else if(options.getInputFormat() == "3D")
		{
			extension = ".3d";
		}
		else if(options.getOutputFormat() == "ALL")
		{
			// Filter supported file formats
			if(it->path().extension() == ".ply" || it->path().extension() == ".txt" || it->path().extension() == ".dat")
			{
				extension = string(it->path().extension().string());
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

    if(options.getStart() <= v.size() && options.getStart() > 0)
    {
        cout << "Starting with scan number " << options.getStart() << endl;
        c = options.getStart() - 1;
    }

    vector<boost::filesystem::path>::iterator endOpt;

    if(options.getEnd() > 0 && options.getEnd() >= options.getStart() && options.getEnd() <= v.size())
    {
        cout << "Ending with scan number " << options.getEnd() << endl;
        endOpt = v.begin() + options.getEnd();
    }
    else
    {
        endOpt = v.end();
    }
   
	for(vector<boost::filesystem::path>::iterator it = v.begin() + c; it != endOpt; it++)
	{
		cout << timestamp << "Converting " << it->string() << endl;
		if(options.getOutputFormat() == "SLAM")
		{
			int reduction = options.getTargetSize();
			ModelPtr model;

			if(reduction == 0 || (reduction != 0 && options.getInputFormat() == "3D"))
			{
				cout << timestamp << "Reading point cloud data from " << it->c_str() << "." << endl;
				model = ModelFactory::readModel(it->string());
                std::string fullFileName = it->c_str();
                
                std::string shortenedFileName = fullFileName.substr(fullFileName.find_last_of("/"), fullFileName.find_last_of(".") - 1);

				if(model)
				{
					char name[1024];
					//sprintf(name, "%s/scan%03d.3d", outputDir.c_str(), c + 1);

                    sprintf(name, "%s/%s.3d", outputDir.c_str(), shortenedFileName.c_str());

					// Check if user wants to reduce. If not, set reduction
					// to 1 to keep all points
					//if(reduction == 0)
					//{
					//	reduction = 1;
					//}

					cout << timestamp << "Saving " << name << "..." << endl;

					ifstream in(it->string().c_str());
					cout << timestamp << "Counting points in " << it->string().c_str() << "..." << endl;

					// Count lines in file
					size_t n_points = 0;
					char line[2048];
					while(in.good())
					{
						in.getline(line, 1024);
						n_points++;
					}
					in.close();

					cout << timestamp << "File " << it->string().c_str() << " contains " << n_points << " points." << endl;

                    int modulo = 1;
                    // If reduction is less than the number of points it will segfault when
                    // calculating the modulo 
                    // and we have to keep all points anyways.
                    // Same if no targetSize was given.
                    if(reduction < n_points && reduction != 0)
                    {
					    modulo = (int)n_points / reduction;
                    }

					ofstream out(name);
					size_t n_ip;
					int cntr = 0;
					floatArr arr = model->m_pointCloud->getPointArray(n_ip);
					for(int a = 0; a < n_ip; a++)
					{
						if(a % modulo == 0)
						{
							if(options.sx() != 1)
							{
								arr[a * 3] 		*= options.sx();
							}

							if(options.sy() != 1)
							{
								arr[a * 3 + 1] 	*= options.sy();
							}

							if(options.sz() != 1)
							{
								arr[a * 3 + 2] 	*= options.sz();
							}

							out << arr[a * 3 + options.x()] << " " << arr[a * 3 + options.y()] << " " << arr[a * 3 + options.z()] << endl;
							cntr++;
						}
					}
					out.close();
					cout << "Wrote " << cntr << " points to file " << name << endl;


				}
			}
			else
			{
				if(options.getInputFormat() == "DAT")
				{
					DatIO io;
					cout << timestamp << "Reading point cloud data from " << it->c_str() << "." << endl;
					model = io.read(it->string(), 4, reduction);

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

					if(reduction == 0)
					{
						char name[1024];
						sprintf(name, "%s/scan%03d.3d", outputDir.c_str(), c);
						cout << timestamp << "Saving " << name << "..." << endl;
						AsciiIO outIO;
						outIO.setModel(model);
						outIO.save(name);
					}
				}

			}

			c++;

		}
		else if(options.getOutputFormat() == "MERGE")
		{
			ModelPtr model = ModelFactory::readModel(it->string());
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


