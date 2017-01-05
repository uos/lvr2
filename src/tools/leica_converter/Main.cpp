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
#include <Eigen/Dense>

using namespace std;

#include <boost/filesystem.hpp>

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_lit.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>

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

namespace qi = boost::spirit::qi;

const leica_convert::Options* options;


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
}

size_t countPointsInFile(boost::filesystem::path& inFile)
{
    ifstream in(inFile.c_str());
    cout << timestamp << "Counting points in " << inFile.filename().string() << "..." << endl;

    // Count lines in file
    size_t n_points = 0;
    char line[2048];
    while(in.good())
    {
        in.getline(line, 1024);
        n_points++;
    }
    in.close();

    cout << timestamp << "File " << inFile.filename().string() << " contains " << n_points << " points." << endl;

    return n_points;
}

size_t writeModel( ModelPtr model,const  boost::filesystem::path& outfile, int modulo)
{
    size_t n_ip;
    size_t cntr = 0;
    floatArr arr = model->m_pointCloud->getPointArray(n_ip);

    size_t new_model_size = n_ip / modulo;
    floatArr targetPoints(new float[3 * new_model_size]);

    for(int a = 0; a < n_ip; a++)
    {
        if(a % modulo == 0)
        {
            if(options->sx() != 1)
            {
                targetPoints[cntr * 3 + options->x()] = arr[a * 3] * options->sx();
            }

            if(options->sy() != 1)
            {
                targetPoints[cntr * 3 + options->y()] = arr[a * 3 + 1] * options->sy();
            }

            if(options->sz() != 1)
            {
                targetPoints[cntr * 3 + options->z()] = arr[a * 3 + 2] * options->sz();
            }
            cntr++;
        }
    }

    PointBufferPtr pc(new PointBuffer);
    pc->setPointArray(targetPoints, new_model_size);
    ModelPtr outModel(new Model(pc));
    ModelFactory::saveModel(outModel, outfile.string());

    return cntr;
}

size_t writeAscii(ModelPtr model, std::ofstream& out, int modulo)
{
    size_t n_ip;
    size_t cntr = 0;
    floatArr arr = model->m_pointCloud->getPointArray(n_ip);
    for(int a = 0; a < n_ip; a++)
    {
        if(a % modulo == 0)
        {
            if(options->sx() != 1)
            {
                arr[a * 3] 		*= options->sx();
            }

            if(options->sy() != 1)
            {
                arr[a * 3 + 1] 	*= options->sy();
            }

            if(options->sz() != 1)
            {
                arr[a * 3 + 2] 	*= options->sz();
            }

            out << arr[a * 3 + options->x()] << " " << arr[a * 3 + options->y()] << " " << arr[a * 3 + options->z()] << endl;
            cntr++;
        }
    }
    return cntr;
}



int asciiReductionFactor(boost::filesystem::path& inFile)
{

    int reduction = options->getTargetSize();

    /*
     * If reduction is less than the number of points it will segfault
     * because the modulo operation is not defined for n mod 0
     * and we have to keep all points anyways.
     * Same if no targetSize was given.
     */
    if(reduction != 0)
    {
        // Count lines in file
        size_t n_points = countPointsInFile(inFile);

        if(reduction < n_points)
        {
            return (int)n_points / reduction;
        }
    }

    /* No reduction write all points */
    return 1;

}

Eigen::Matrix4d buildTransformation(double* alignxf) 
{
    Eigen::Matrix3d rotation;
    Eigen::Vector4d translation;

    rotation  << alignxf[0],  alignxf[4],  alignxf[8],
    alignxf[1],  alignxf[5],  alignxf[9],
    alignxf[2],  alignxf[6],  alignxf[10];

    translation << alignxf[12], alignxf[13], alignxf[14], 1.0;

    Eigen::Matrix4d transformation;
    transformation.setIdentity();
    transformation.block<3,3>(0,0) = rotation;
    transformation.rightCols<1>() = translation;

    return transformation;
}

Eigen::Matrix4d getTransformationFromPose(boost::filesystem::path& pose)
{
    ifstream poseIn(pose.c_str());
    if(poseIn.good())
    {
        double rPosTheta[3];
        double rPos[3];
        double alignxf[16];

        poseIn >> rPos[0] >> rPos[1] >> rPos[2];
        poseIn >> rPosTheta[0] >> rPosTheta[1] >> rPosTheta[2];

        rPosTheta[0] *= 0.0174533;
        rPosTheta[1] *= 0.0174533;
        rPosTheta[2] *= 0.0174533;

        double sx = sin(rPosTheta[0]);
        double cx = cos(rPosTheta[0]);
        double sy = sin(rPosTheta[1]);
        double cy = cos(rPosTheta[1]);
        double sz = sin(rPosTheta[2]);
        double cz = cos(rPosTheta[2]);

        alignxf[0]  = cy*cz;
        alignxf[1]  = sx*sy*cz + cx*sz;
        alignxf[2]  = -cx*sy*cz + sx*sz;
        alignxf[3]  = 0.0;
        alignxf[4]  = -cy*sz;
        alignxf[5]  = -sx*sy*sz + cx*cz;
        alignxf[6]  = cx*sy*sz + sx*cz;
        alignxf[7]  = 0.0;
        alignxf[8]  = sy;
        alignxf[9]  = -sx*cy;
        alignxf[10] = cx*cy;

        alignxf[11] = 0.0;

        alignxf[12] = rPos[0];
        alignxf[13] = rPos[1];
        alignxf[14] = rPos[2];
        alignxf[15] = 1;

        //		Eigen::Matrix4d transformation;
        //		transformation  << alignxf[0],  alignxf[1],  alignxf[2],  alignxf[3],
        //				alignxf[4],  alignxf[5],  alignxf[6],  alignxf[7],
        //				alignxf[8],  alignxf[9],  alignxf[10], alignxf[11],
        //				alignxf[12], alignxf[13], alignxf[14], alignxf[15];
        /*
           Eigen::Matrix3d rotation;
           Eigen::Vector4d translation;

           rotation  << alignxf[0],  alignxf[4],  alignxf[8],
           alignxf[1],  alignxf[5],  alignxf[9],
           alignxf[2],  alignxf[6],  alignxf[10];

           translation << alignxf[12], alignxf[13], alignxf[14], 1.0;

           Eigen::Matrix4d transformation;
           transformation.setIdentity();
           transformation.block<3,3>(0,0) = rotation;
           transformation.rightCols<1>() = translation; */


        return buildTransformation(alignxf);
    }
    else
    {
        return Eigen::Matrix4d::Identity();
    }
}

Eigen::Matrix4d getTransformationFromFrames(boost::filesystem::path& frames)
{
    double alignxf[16];
    int color;

    std::ifstream in(frames.c_str());
    int c = 0;
    while(in.good())
    {
        c++;
        for(int i = 0; i < 16; i++)
        {
            in >> alignxf[i];
        }

        in >> color;

        if(!in.good())
        {
            c = 0;
            break;
        }
    }

    /*
       Eigen::Matrix3d rotation;
       Eigen::Vector4d translation;

       rotation  << alignxf[0],  alignxf[4],  alignxf[8],
       alignxf[1],  alignxf[5],  alignxf[9],
       alignxf[2],  alignxf[6],  alignxf[10];

       translation << alignxf[12], alignxf[13], alignxf[14], 1.0;

       Eigen::Matrix4d transformation;
       transformation.setIdentity();
       transformation.block<3,3>(0,0) = rotation;
       transformation.rightCols<1>() = translation; */

    return buildTransformation(alignxf);
}

void transformModel(ModelPtr model, Eigen::Matrix4d transformation)
{
    cout << timestamp << "Transforming model." << endl;
    size_t numPoints;
    floatArr arr = model->m_pointCloud->getPointArray(numPoints);

    for(int i = 0; i < numPoints; i++)
    {
        float x = arr[3 * i];
        float y = arr[3 * i + 1];
        float z = arr[3 * i + 2];

        Eigen::Vector4d v(x,y,z,1);
        Eigen::Vector4d tv = transformation * v;

        arr[3 * i]     = tv[0];
        arr[3 * i + 1] = tv[1];
        arr[3 * i + 2] = tv[2];
    }
}

void processSingleFile(boost::filesystem::path& inFile)
{
    cout << timestamp << "Processing " << inFile << endl;

    ModelPtr model;

    cout << timestamp << "Reading point cloud data from file " << inFile.filename().string() << "." << endl;

    model = ModelFactory::readModel(inFile.string());


    if(options->getOutputFile() != "")
    {
        // Merge (only ASCII)
        char frames[1024];
        char pose[1024];
        sprintf(frames, "%s/%s.frames", inFile.parent_path().c_str(), inFile.stem().c_str());
        sprintf(pose, "%s/%s.pose", inFile.parent_path().c_str(), inFile.stem().c_str());

        boost::filesystem::path framesPath(frames);
        boost::filesystem::path posePath(pose);

        if(boost::filesystem::exists(framesPath))
        {
            std::cout << timestamp << "Getting transformation from frame: " << framesPath << std::endl;
            Eigen::Matrix4d transform = getTransformationFromFrames(framesPath);
            transformModel(model, transform);
        }
        else if(boost::filesystem::exists(posePath))
        {

            std::cout << timestamp << "Getting transformation from pose: " << posePath << std::endl;
            Eigen::Matrix4d transform = getTransformationFromPose(posePath);
            transformModel(model, transform);
        }

        static size_t points_written = 0;

        std::ofstream out;

        /* If points were written we want to append the next scans, otherwise we want an empty file */
        if(points_written != 0)
        {
            out.open(options->getOutputFile().c_str(), std::ofstream::out | std::ofstream::app);
        }
        else
        {
            out.open(options->getOutputFile().c_str(), std::ofstream::out | std::ofstream::trunc);
        }

        points_written += writeAscii(model, out, asciiReductionFactor(inFile));

        out.close();
    }
    else
    {
        if(options->getOutputFormat() == "")
        {
            // Infer format from file extension, convert and write out
        }
        else if(options->getOutputFormat() == "SLAM")
        {
            // Transform and write in slam format
            static int n = 0;

            char name[1024];
            char pose[1024];

            sprintf(name, "/%s/scan%3d.3d", options->getOutputDir().c_str(), n);
            sprintf(name, "/%s/scan%3d.pose", options->getOutputDir().c_str(), n);

            ofstream poseOut(pose);

            // TO-DO Pose or frame existing
            poseOut << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " "<< 0 << endl;
            poseOut.close();

            ofstream out(name);
            size_t points_written = writeAscii(model, out, asciiReductionFactor(inFile));

            out.close();
            cout << "Wrote " << points_written << " points to file " << name << endl;
            n++;
        }
        else
        {
            // Transform and write to target format.
            char frames[1024];
            char pose[1024];
            char outFile[1024];

            sprintf(outFile, "/%s/%s", options->getOutputDir().c_str(), inFile.filename().c_str());
            sprintf(frames, "/%s/%s.frames", inFile.parent_path().c_str(), inFile.stem().c_str());
            sprintf(pose, "/%s/%s.pose", inFile.parent_path().c_str(), inFile.stem().c_str());

            boost::filesystem::path framesPath(frames);
            boost::filesystem::path posePath(pose);

            if(boost::filesystem::exists(framesPath))
            {
                Eigen::Matrix4d transform = getTransformationFromFrames(framesPath);
                transformModel(model, transform);
            }
            else if(boost::filesystem::exists(posePath))
            {
                Eigen::Matrix4d transform = getTransformationFromPose(posePath);
                transformModel(model, transform);
            }


            static size_t points_written = writeModel(model, boost::filesystem::path(outFile), asciiReductionFactor(inFile));

        }
    }



    //	if(options->slamOut())
    //	{
    //		if(model)
    //		{
    //
    //		}
    //	}
    //	else
    //	{
    //		if(options->getOutputFile() != "")
    //		{
    //
    //		}
    //		else
    //		{
    //			if(options->getOutputFormat() == "")
    //			{
    //
    //				// TO-DO Test if outputdir is inputdir
    //
    //
    //			}
    //			else if(options->getOutputFormat() == "ASCII")
    //			{
    //				// Write all data into points.txt
    //				char frames[1024];
    //				char pose[1024];
    //				sprintf(frames, "/%s/%s.frames", inFile.parent_path().c_str(), inFile.stem().c_str());
    //				sprintf(pose, "/%s/%s.pose", inFile.parent_path().c_str(), inFile.stem().c_str());
    //
    //				boost::filesystem::path framesPath(frames);
    //				boost::filesystem::path posePath(pose);
    //
    //				if(boost::filesystem::exists(framesPath))
    //				{
    //					Eigen::Matrix4d transform = getTransformationFromFrames(framesPath);
    //					transformModel(model, transform);
    //				}
    //				else if(boost::filesystem::exists(posePath))
    //				{
    //					Eigen::Matrix4d transform = getTransformationFromPose(posePath);
    //					transformModel(model, transform);
    //				}
    //
    //				std::ofstream out("points.txt", std::ofstream::out | std::ofstream::app);
    //
    //				static size_t points_written = 0;
    //				points_written += writeModel(model, out, asciiReductionFactor(inFile));
    //
    //				out.close();
    //
    //			}
    //		}
    //	}


    //		if(options->getInputFormat() == "DAT")
    //		{
    //			DatIO io;
    //			cout << timestamp << "Reading point cloud data from " << it->c_str() << "." << endl;
    //			model = io.read(it->string(), 4, reduction);
    //
    //			if(options->filter())
    //			{
    //				cout << timestamp << "Filtering input data..." << endl;
    //				model = filterModel(model, options->getK(), options->getSigma());
    //			}
    //		}
    //		else
    //		{
    //			cout << timestamp << "Reduction mode currently only supported for DAT format." << endl;
    //			exit(-1);
    //		}
    //
    //		if(model)
    //		{
    //			/*	// Convert to slam coordinate system
    //				if(model->m_pointCloud)
    //				{
    //					float point[3];
    //					PointBufferPtr p_ptr = model->m_pointCloud;
    //					size_t num;
    //					floatArr points = p_ptr->getPointArray(num);
    //					for(int i = 0; i < num; i++)
    //					{
    //						point[0] = points[3 * i + 1];
    //						point[1] = points[3 * i + 2];
    //						point[2] = points[3 * i];
    //
    //						point[0] *= -100;
    //						point[1] *= 100;
    //						point[2] *= 100;
    //
    //						points[3 * i] = point[0];
    //						points[3 * i + 1] = point[1];
    //						points[3 * i + 2] = point[2];
    //					}
    //				}
    //			 */
    //
    //			if(reduction == 0)
    //			{
    //				char name[1024];
    //				sprintf(name, "%s/scan%03d.3d", outputDir.c_str(), c);
    //				cout << timestamp << "Saving " << name << "..." << endl;
    //				AsciiIO outIO;
    //				outIO.setModel(model);
    //				outIO.save(name);
    //			}
    //		}
    //	}
    //	else if(options->getOutputFormat() == "MERGE")
    //	{
    //		ModelPtr model = ModelFactory::readModel(it->string());
    //		if(model)
    //		{
    //			PointBufferPtr points = model->m_pointCloud;
    //			size_t num_points = 0;
    //			size_t num_colors = 0;
    //			floatArr point_arr = points->getPointArray(num_points);
    //			ucharArr color_arr = points->getPointColorArray(num_colors);
    //
    //			cout << timestamp << "Adding " << it->c_str() << " to merged point cloud" << endl;
    //
    //			for(size_t i = 0; i < num_points; i++)
    //			{
    //				merge_points.push_back(point_arr[3 * i]);
    //				merge_points.push_back(point_arr[3 * i + 1]);
    //				merge_points.push_back(point_arr[3 * i + 2]);
    //
    //				if(num_points == num_colors)
    //				{
    //					merge_colors.push_back(color_arr[3 * i]);
    //					merge_colors.push_back(color_arr[3 * i + 1]);
    //					merge_colors.push_back(color_arr[3 * i + 2]);
    //				}
    //				else
    //				{
    //					for(int j = 0; j < 3; j++)
    //					{
    //						merge_colors.push_back(128);
    //					}
    //				}
    //			}
    //		}
    //		else
    //		{
    //			cout << "Unable to model data from " << it->c_str() << endl;
    //		}
    //	}

}

bool parse_filename(std::string::iterator first, std::string::iterator last, int& i)
{   

    using qi::lit;
    using qi::uint_parser;
    using qi::parse;
    using boost::spirit::qi::_1;
    using boost::phoenix::ref;

    uint_parser<unsigned, 10, 3, 3> uint_3_d;

    bool r = parse(
            first,                          /*< start iterator >*/
            last,                           /*< end iterator >*/
            ((lit("scan")|lit("Scan")) >> uint_3_d[ref(i) = _1])   /*< the parser >*/
            );

    std::cout << "Number: " <<  i << std::endl;
    if (first != last) // fail if we did not get a full match
        return false;
    return r;
}

bool sortScans(boost::filesystem::path firstScan, boost::filesystem::path secScan)
{
    std::string firstStem = firstScan.stem().string();
    std::string secStem   = secScan.stem().string();

    int i = 0;
    int j = 0;

    bool first = parse_filename(firstStem.begin(), firstStem.end(), i);
    bool sec = parse_filename(secStem.begin(), secStem.end(), j);

    if(first && sec)
    {
        return (i < j);
    }
    else
    {
        if(!first)
        {
            std::cerr << timestamp << " " << firstScan << " does not match the naming convention" << std::endl;
            std::terminate();
        }
        else
        {
            std::cerr << "ERROR: " << timestamp << " " << secScan << " does not match the naming convention" << std::endl;
            std::terminate();
        }
    }

}

int main(int argc, char** argv) {
    // Parse command line arguments
    options = new leica_convert::Options(argc, argv);

    boost::filesystem::path inputDir(options->getInputDir());
    boost::filesystem::path outputDir(options->getOutputDir());

    // Check input directory
    if(!boost::filesystem::exists(inputDir))
    {
        cout << timestamp << "Error: Directory " << options->getInputDir() << " does not exist" << endl;
        exit(-1);
    }

    // Check if output dir exists
    if(!boost::filesystem::exists(outputDir))
    {
        cout << timestamp << "Creating directory " << options->getOutputDir() << endl;
        if(!boost::filesystem::create_directory(outputDir))
        {
            cout << timestamp << "Error: Unable to create " << options->getOutputDir() << endl;
            exit(-1);
        }
    }

    boost::filesystem::path abs_in = boost::filesystem::canonical(inputDir);
    boost::filesystem::path abs_out = boost::filesystem::canonical(outputDir);

    if(abs_in == abs_out)
    {
        cout << timestamp << "Error: We think it is not a good idea to write into the same directory. " << endl;
        exit(-1);
    }

    // Create director iterator and parse supported file formats
    boost::filesystem::directory_iterator end;
    vector<boost::filesystem::path> v;
    for(boost::filesystem::directory_iterator it(inputDir); it != end; ++it)
    {
        std::string ext =	it->path().extension().string();
        if(ext == ".3d" || ext == ".ply" || ext == ".dat" || ext == ".txt" )
        {
            v.push_back(it->path());
        }
    }

    // Sort entries
    sort(v.begin(), v.end(), sortScans);

    vector<float>	 		merge_points;
    vector<unsigned char>	merge_colors;

    int j = -1;
    for(vector<boost::filesystem::path>::iterator it = v.begin(); it != v.end(); ++it)
    {
        int i = 0;

        std::string currFile = (it->stem()).string();
        bool p = parse_filename(currFile.begin(), currFile.end(), i);

        //if parsing failed terminate, this should never happen.
        if(!p)
        {
            std::cerr << "ERROR " << timestamp << " " << *it << " does not match the naming convention" << std::endl;
            break;
        }

        // check if the current scan has the same numbering like the previous, this should not happen.
        if(i == j)
        {
            std::cerr << "ERROR " << timestamp << " " << *std::prev(it) << " & " << *it << " have identical numbering" << std::endl;
            break;
        }

        // check if the scan is in the range which should be processed
        if(i >= options->getStart()){
            if(i <= options->getEnd())
            {
                processSingleFile(*it);
                j = i;
            }
            else
            {
                break;
            }
        }

    }

    cout << timestamp << "Program end." << endl;
    delete options;
    return 0;
}
