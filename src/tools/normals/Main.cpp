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


// Program options for this tool
#include "Options.hpp"
#include "io/AsciiIO.hpp"
#include "io/Timestamp.hpp"
#include "io/Progress.hpp"
#include "io/DataStruct.hpp"
#include "io/ModelFactory.hpp"
#include "geometry/Matrix4.hpp"
#include "geometry/Normal.hpp"
#include "reconstruction/SearchTree.hpp"
#include "reconstruction/SearchTreeStann.hpp"

// SearchTreePCL
#ifdef _USE_PCL_
    #include "reconstruction/SearchTreeFlann.hpp"
#endif

#include <boost/filesystem.hpp>

#include <iostream>
#include <string>
using std::cout;
using std::endl;
using std::string;

using namespace lvr;

Matrix4<float> readFrameFile(string filename)
{
    ifstream frameFile(filename);
    float m[16], color;
    while(frameFile.good())
    {
        for(int i = 0; i < 16; i++ && frameFile.good()) frameFile >> m[i];
        frameFile >> color;
    }

    return Matrix4<float>(m);
}

Matrix4<float> readPoseFile(string filename)
{
    float euler[6] = {0};
    ifstream poseFile(filename);
    if(poseFile.good())
    {
       for(int i = 0; i < 6; i++)
       {
           poseFile >> euler[i];
       }
    }
    Vertex<float> position(euler[0], euler[1], euler[2]);
    Vertex<float> angle(euler[3] * 0.017453293, euler[4] * 0.017453293, euler[5] * 0.017453293);

    Matrix4<float> trans(position, angle);
    return trans;
}

void dumpNormals(string filename, PointBufferPtr ptr)
{
    ofstream out(filename.c_str());
    size_t count;
    floatArr normals = ptr->getPointNormalArray(count);
    for(size_t i = 0; i < count; i++)
    {
        size_t j = 3 * i;
        out << normals[j] << " " << normals[j + 1] << " " << normals[j + 2] << endl;
    }
    out.close();
}

void interpolateNormals(PointBufferPtr pc, size_t numPoints, int n)
{
    cout << timestamp << "Creating search tree for interpolation" << endl;

    SearchTree<Vertex<float> >::Ptr       tree;
#ifdef _USE_PCL_
        tree = SearchTree<Vertex<float> >::Ptr( new SearchTreeFlann<Vertex<float> >(pc, numPoints, n, n, n) );
#else
        cout << timestamp << "Warning: PCL is not installed. Using STANN search tree in AdaptiveKSearchSurface." << endl;
        tree = SearchTree<Vertex<float> >::Ptr( new SearchTreeStann<Vertex<float> >(pc, numPoints, n, n, n) );
#endif

    size_t count;
    floatArr points = pc->getPointArray(count);
    floatArr normals = pc->getPointNormalArray(count);

    string comment = timestamp.getElapsedTime() + "Interpolating normals ";
    ProgressBar progress(numPoints, comment);

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < numPoints; i++)
    {
        // Create search tree
        vector< ulong > indices;
        vector< double > distances;

        Vertex<float> vertex(points[3 * i], points[3 * i + 1], points[3 * i + 2]);
        tree->kSearch( vertex, n, indices, distances);

        // Do interpolation
        Normal<float> normal(normals[3 * i], normals[3 * i + 1], normals[3 * i + 2]);
        for(int j = 0; j < indices.size(); j++)
        {
            normal += Normal<float>(normals[3 * indices[j]], normals[3 * indices[j] + 1], normals[3 * indices[j] + 2]);
        }
        normal.normalize();

        // Save results in buffer (I know i should use a seperate buffer, but for testing it
        // should be OK to save the interpolated values directly into the input buffer)
        normals[3 * i]      = normal.x;
        normals[3 * i + 1]  = normal.y;
        normals[3 * i + 2]  = normal.z;

        ++progress;
    }
    cout << endl;
}

/**
 * @brief   Main entry point for the LSSR surface executable
 */
int main(int argc, char** argv)
{
	try
	{
	    normals::Options options(argc, argv);

	    cout << options << endl;

	    string dir = options.getInputDirectory();
	    string outFileName = options.getOutputFile();
	    int start = options.getStart();
	    int end = options.getEnd();
	    int target = options.getTargetSize();

	    // Try to access directory
	    boost::filesystem::path directory(dir);
	    if(is_directory(directory))
	    {
	        // Iterate through directory, count relevant objects
	        int n3dFiles = 0;

	        // First and last scan to load
	        int firstScan = -1;
	        int lastScan =  -1;

	        boost::filesystem::directory_iterator lastFile;

	        // First, look for .3d files
	        for(boost::filesystem::directory_iterator it(directory); it != lastFile; it++ )
	        {
	            boost::filesystem::path p = it->path();
	            if(string(p.extension().string().c_str()) == ".3d")
	            {
	                // Check for naming convention "scanxxx.3d"
	                int num = 0;
	                if(sscanf(p.filename().string().c_str(), "scan%3d", &num))
	                {
	                    n3dFiles++;
	                    if(firstScan == -1) firstScan = num;
	                    if(lastScan == -1) lastScan = num;

	                    if(num > lastScan) lastScan = num;
	                    if(num < firstScan) firstScan = num;
	                }
	            }
	        }

	        if(firstScan == -1 && lastScan == -1)
	        {
	            cout << timestamp << "Directory " << directory << " contains no scan file. Exiting..." << endl;
	            return 0;
	        }


	        // Trim Range
	        if(end < lastScan && end >= 0 && end >= firstScan)
	        {
	            lastScan = end;
	        }

	        if(start >= firstScan && firstScan >= 0 && start <= end)
	        {
	            firstScan = start;
	        }

	        // Count total number of points in scan range
	        size_t totalPointCount = 0;
	        for(int i = firstScan; i <= lastScan; i++)
	        {
	            char scanFileName[100];
	            sprintf(scanFileName, "scan%03d.3d", i);
	            boost::filesystem::path filePath(scanFileName);
	            boost::filesystem::path scanFilePath = directory / filePath;

	            string scanFile = scanFilePath.string().c_str();
	            cout << timestamp << "Counting points in file " << scanFile << std::flush;
	            size_t pointsInScan = AsciiIO::countLines(scanFile);
	            cout << " --> " << pointsInScan << endl;
	            totalPointCount += pointsInScan;
	        }
	        cout << timestamp << "Total number of points to process: " << totalPointCount << endl;

	        // Calc points to read based on required reduction
	        size_t skipPoints = 1;
	        size_t numPointsToRead = totalPointCount;
	        if(target > 0)
	        {
	           skipPoints = (int)totalPointCount / target;
	           numPointsToRead = target;

	        }

	        // Create buffer arrays
	        floatArr points(new float[3 * numPointsToRead]);
	        floatArr normals(new float[3 * numPointsToRead]);


	        cout << timestamp << "Reducing number of points to " << target << ". Writing every " << skipPoints << "th point." << endl;
	        size_t pointsRead = 0;
	        size_t counter;
	        for(int i = firstScan; i <= lastScan; i++)
	        {
	            char scanFileName[100];
	            sprintf(scanFileName, "scan%03d.3d", i);
	            boost::filesystem::path filePath(scanFileName);
	            boost::filesystem::path scanFilePath = directory / filePath;

	            char poseFileName[100];
	            sprintf(poseFileName, "scan%03d.pose", i);
	            boost::filesystem::path posePath(poseFileName);
	            boost::filesystem::path poseFilePath = directory / posePath;

	            char frameFileName[100];
	            sprintf(frameFileName, "scan%03d.frames", i);
	            boost::filesystem::path framePath(frameFileName);
	            boost::filesystem::path frameFilePath = directory / framePath;

	            // Get transformation from frames or pose files if possible
	            Matrix4<float> transform;
	            if(exists(frameFilePath))
	            {
	                float pose[6];
	                cout << timestamp << "Reading " << frameFilePath.c_str() << std::flush;
	                transform = readFrameFile(frameFilePath.string().c_str());
	                transform.toPostionAngle(pose);
	                cout << " --> " << pose[0] << " " << pose[1] << " " << pose[2] << " ";
	                cout << pose[3] << " " << pose[4] << " " << pose[5] << " " << endl;
 	            }
	            else if(exists(poseFilePath))
	            {
	                float pose[6];
	                cout << timestamp << "Reading " << poseFilePath.c_str() << std::flush;
	                transform = readPoseFile(poseFilePath.string().c_str());
	                transform.toPostionAngle(pose);
	                cout << " --> " << pose[0] << " " << pose[1] << " " << pose[2] << " ";
	                cout << pose[3] << " " << pose[4] << " " << pose[5] << " " << endl;
	            }
	            else
	            {
	                cout << timestamp << "Warning: No transformation found for scan " << i << ". Copying points." << endl;
	            }

	            string scanFile = scanFilePath.string().c_str();
	            cout << timestamp << "Processing " << scanFile << endl;

	            // Open scan file
	            ifstream in(scanFilePath.c_str());
	            float x, y, z, nx, ny, nz;
	            do
	            {
	                in >> x >> y >> z >> nx >> ny >> nz;
	                if(counter % skipPoints == 0)
	                {
	                    // Transform normal according to pose
	                    Normal<float> normal(nx, ny, nz);
	                    Vertex<float> point(x, y, z);
	                    normal = transform * normal;
	                    point = transform * point;

	                    // Write data into buffer
	                    points[pointsRead * 3]     = point.x;
	                    points[pointsRead * 3 + 1] = point.y;
	                    points[pointsRead * 3 + 2] = point.z;

	                    normals[pointsRead * 3]     = normal.x;
	                    normals[pointsRead * 3 + 1] = normal.y;
	                    normals[pointsRead * 3 + 2] = normal.z;
	                    pointsRead++;
	                }
	                counter++;
	            } while(in.good() && pointsRead < numPointsToRead);

	        }
	        cout << timestamp << "Read " << pointsRead << " from " << numPointsToRead << " requested." << endl;


	        PointBufferPtr pc = PointBufferPtr( new PointBuffer );
	        pc->setPointArray(points, numPointsToRead);
	        pc->setPointNormalArray(normals, numPointsToRead);

	        if(options.getInterpolation() > 0)
	        {
	            interpolateNormals(pc, numPointsToRead, options.getInterpolation());
	        }
	        ModelPtr model(new Model(pc));

            cout << timestamp << "Writing " << outFileName << endl;
	        ModelFactory::saveModel(model, outFileName);
	    }
	}
	catch(...)
	{
		std::cout << "Unable to parse options. Call 'lvr_registration --help' for more information." << std::endl;
	}
	return 0;
}

