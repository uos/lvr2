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

/**
 * UosIO.tcc
 *
 *  @date 11.05.2011
 *  @author Thomas Wiemann
 */

#include <list>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>

using std::list;
using std::vector;
using std::ifstream;
using std::stringstream;

#include <boost/filesystem.hpp>

#include "lvr2/io/UosIO.hpp"
#include "lvr2/io/Progress.hpp"
#include "lvr2/io/Timestamp.hpp"

namespace lvr2
{


ModelPtr UosIO::read(string dir)
{
    ModelPtr model;

    size_t n = 0;
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
            if(p.extension().string() == ".3d")
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

        // Check if given directory contains scans in new format.
        // If so, read them and return result. Otherwise try to
        // read new format.
        if(n3dFiles > 0)
        {
            // Check for user scan ranges
            if(m_firstScan > -1 && m_firstScan <= lastScan)
            {
                firstScan = m_firstScan;
            }

            if(m_lastScan >= -1 && m_lastScan <= lastScan && m_lastScan >= firstScan)
            {
                lastScan = m_lastScan;
            }

            m_firstScan = firstScan;
            m_lastScan = lastScan;

            cout << timestamp << "Reading " << n3dFiles << " scans in UOS format "
                << "(From " << firstScan << " to " << lastScan << ")." << endl;
            readNewFormat(model, dir, firstScan, lastScan, n);
        }
        else
        {
            // Count numbered sub directories, ignore others
            int nDirs = 0;
            for(boost::filesystem::directory_iterator it(directory); it != lastFile; it++ )
            {
                boost::filesystem::path p = it->path();
                int num = 0;

                // Only count numbered dirs
                if(sscanf(p.filename().string().c_str(), "%d", &num))
                {
                    if(firstScan == -1) firstScan = num;
                    if(lastScan == -1) lastScan = num;


                    if(num > lastScan) lastScan = num;
                    if(num < firstScan) firstScan = num;

                    nDirs++;
                }
            }

            // Check is dirs were found and try to read old format
            if(nDirs)
            {
                // Check for user scan ranges
                if(m_firstScan > -1 && m_firstScan <= lastScan)
                {
                    firstScan = m_firstScan;
                }

                if(m_lastScan >= -1 && m_lastScan <= lastScan && m_lastScan >= firstScan)
                {
                    lastScan = m_lastScan;
                }

                m_firstScan = firstScan;
                m_lastScan = lastScan;

                cout << timestamp << "Reading " << nDirs << " scans in old UOS format "
                    << "(From " << firstScan << " to " << lastScan << ")." << endl;
                readOldFormat(model, dir, firstScan, lastScan, n);
            }
            else
            {
                return ModelPtr();
            }
        }

    }
    else
    {
        cout << timestamp << "UOSReader: " << dir << " is not a directory." << endl;
    }

    m_model = model;
    return model;
}


void UosIO::reduce(string dir, string target, int reduction)
{
    // Open output stream
    m_outputFile.open(target.c_str());
    if(!m_outputFile.good())
    {
        cout << timestamp << "UOSReader: " << dir << " unable to open " << target << " for writing." << endl;
        return;
    }

    // Set needed flags for inout code
    m_reductionTarget = reduction;
    m_saveToDisk = true;

    // Read data and write reduced points
    ModelPtr m = read(dir);


}


void UosIO::readNewFormat(ModelPtr &model, string dir, int first, int last, size_t &n)
{
    list<Vec > allPoints;
    list<BaseVector<int> > allColors;

    size_t point_counter = 0;

    vector<indexPair> sub_clouds;

    // Count points in all given files
    size_t numPointsTotal = 0;
    for(int fileCounter = first; fileCounter <= last; fileCounter++)
    {
        // Create scan file name
        boost::filesystem::path scan_path(
                boost::filesystem::path(dir) /
                boost::filesystem::path( "scan" + to_string( fileCounter, 3 ) + ".3d" ) );
        string scanFileName = "/" + scan_path.relative_path().string();

        // Count lines in scan
        numPointsTotal += AsciiIO::countLines(scanFileName);
    }

    // Calculate the number of points to skip when writing to disk
    size_t skipPoints = 1;

    if(m_reductionTarget > 1)
    {
        skipPoints = (int)numPointsTotal / m_reductionTarget;
    }

    if(m_saveToDisk)
    {
        cout << timestamp << "Reduction mode. Writing every " << skipPoints << "th point." << endl;
    }

    for(int fileCounter = first; fileCounter <= last; fileCounter++)
    {
        // New (unit) transformation matrix
        Matrix4<Vec> tf;

        // Input file streams for scan data, poses and frames
        ifstream scan_in, pose_in, frame_in;

        // Create scan file name
        boost::filesystem::path scan_path(
                boost::filesystem::path(dir) /
                boost::filesystem::path( "scan" + to_string( fileCounter, 3 ) + ".3d" ) );
        string scanFileName = scan_path.string();

        // Count lines in scan
        size_t points_in_file = AsciiIO::countLines(scanFileName);

        int num_attributes = AsciiIO::getEntriesInLine(scanFileName) - 3;
        bool has_color = (num_attributes == 3) || (num_attributes == 4);
        bool has_intensity = (num_attributes == 1) || (num_attributes == 4);

        if(has_color)
        {
            cout << timestamp << "Reading color information." << endl;
        }

        if(has_intensity)
        {
            cout << timestamp << "Reading intensity information." << endl;
        }

        // Read scan data
        scan_in.open(scanFileName.c_str());
        if(!scan_in.good())
        {
            // Continue with next file if the expected file couldn't be read
            cout << timestamp << "UOS Reader: Unable to read scan " << scanFileName << endl;
            scan_in.close();
            scan_in.clear();
            continue;
        }
        else
        {
            // Tmp list of read points
            list<Vec > tmp_points;


            // Try to get fransformation from .frames file
            boost::filesystem::path frame_path(
                    boost::filesystem::path(dir) /
                    boost::filesystem::path( "scan" + to_string( fileCounter, 3 ) + ".frames" ) );
            string frameFileName = frame_path.string();

            frame_in.open(frameFileName.c_str());
            if(!frame_in.good())
            {
                // Try to parse .pose file
                boost::filesystem::path pose_path(
                        boost::filesystem::path(dir) /
                        boost::filesystem::path( "scan" + to_string( fileCounter, 3 ) + ".pose" ) );
                string poseFileName = pose_path.string();

                pose_in.open(poseFileName.c_str());
                if(pose_in.good())
                {
                    float euler[6];
                    for(int i = 0; i < 6; i++) pose_in >> euler[i];

                    euler[3] *= 0.017453293;
                    euler[4] *= 0.017453293;
                    euler[5] *= 0.017453293;

                    Vec position(euler[0], euler[1], euler[2]);
                    Vec angle(euler[3], euler[4], euler[5]);

                    tf = Matrix4<Vec>(position, angle);
                }
                else
                {
                    cout << timestamp << "UOS Reader: Warning: No position information found." << endl;
                    tf = Matrix4<Vec>();
                }

            }
            else
            {
                // Use transformation from .frame files
                tf = parseFrameFile(frame_in);

            }

            // Print pose information
            float euler[6];
            tf.toPostionAngle(euler);

            cout << timestamp << "Processing " << scanFileName << " @ "
                << euler[0] << " " << euler[1] << " " << euler[2] << " "
                << euler[3] << " " << euler[4] << " " << euler[5] << endl;

            // Skip first line in scan file (maybe metadata)
            char dummy[1024];
            scan_in.getline(dummy, 1024);

            // Setup info output
            string comment = timestamp.getElapsedTime() + "Reading file " + scanFileName;
            ProgressBar progress(points_in_file, comment);

            // Read all points
            while(scan_in.good())
            {
                float x, y, z, rem, dummy;
                int r, g, b;

                point_counter ++;

                if(has_intensity && !has_color)
                {
                    scan_in >> x >> y >> z >> rem;
                }
                else if(has_intensity && has_color)
                {
                    scan_in >> x >> y >> z >> rem >> r >> g >> b;
                    allColors.push_back(BaseVector<int> (r, g, b));
                }
                else if(has_color && !has_intensity)
                {
                    scan_in >> x >> y >> z >> r >> g >> b;
                    allColors.push_back(BaseVector<int> (r, g, b));
                }
                else
                {
                    scan_in >> x >> y >> z;
                    for(int n_dummys = 0; n_dummys < num_attributes; n_dummys++) scan_in >> dummy;
                }

                BaseVector<float> point(x, y, z);
                BaseVector<unsigned char> color;

                // Code branching for point converter!
                if(!m_saveToDisk)
                {
                    tmp_points.push_back(point);
                }
                else
                {
                    if(m_outputFile.good())
                    {
                        if(point_counter % skipPoints == 0)
                        {
                            point = tf * point;
                            m_outputFile << point[0] << " " << point[1] << " " << point[2] << " ";

                            // Save remission values if present
                            if(has_intensity && m_saveRemission)
                            {
                                m_outputFile << rem << " ";
                            }

                            // Save color values if present
                            if(has_color)
                            {
                                m_outputFile << r << " " << g << " " << b;
                            }
                            else if(m_saveRemissionColor)
                            {
                                r = g = b = rem;
                                m_outputFile << r << " " << g << " " << b;
                            }
                            m_outputFile << endl;
                        }
                    }
                }
                ++progress;
            }

            // Save index of first point of new scan
            size_t firstIndex;
            if(allPoints.size() > 0)
            {
                firstIndex = allPoints.size();
            }
            else
            {
                firstIndex = 0;
            }

            // Transform scan point with current matrix
            list<Vec >::iterator it, it1;
            for(it = tmp_points.begin(); it != tmp_points.end(); it++)
            {
                Vec v = *it;
                v = tf * v;
                allPoints.push_back(v);
            }

            // Save last index
            size_t lastIndex;
            if(allPoints.size() > 0)
            {
                lastIndex = allPoints.size() - 1;
            }
            else
            {
                lastIndex = 0;
            }

            // Save index pair for current scan
            sub_clouds.push_back(make_pair(firstIndex, lastIndex));
            m_numScans++;
        }
        cout << endl;
    }

    // Convert into array
    if ( allPoints.size() )
    {
        cout << timestamp << "UOS Reader: Read " << allPoints.size() << " points." << endl;

        // Save position information

        size_t numPoints(0);
        floatArr points;
        ucharArr pointColors;


        numPoints = allPoints.size();
        points = floatArr( new float[3 * allPoints.size()] );
        list<Vec >::iterator p_it;
        size_t i(0);
        for( p_it = allPoints.begin(); p_it != allPoints.end(); p_it++ )
        {
            Vec v = *p_it;
            points[i    ] = v[0];
            points[i + 1] = v[1];
            points[i + 2] = v[2];
            i += 3;
        }

        // Save color information
        if ( allColors.size() )
        {
            pointColors = ucharArr( new unsigned char[ 3 * numPoints ] );
            i = 0;
            list<BaseVector<int>>::iterator c_it;
            for(c_it = allColors.begin(); c_it != allColors.end(); c_it++)
            {
                BaseVector<int> v = *c_it;
                pointColors[i    ] = (unsigned char) v[0];
                pointColors[i + 1] = (unsigned char) v[1];
                pointColors[i + 2] = (unsigned char) v[2];
                i += 3;
            }
        }

        // Create point cloud in model
        model = ModelPtr( new Model );
        model->m_pointCloud = PointBufferPtr( new PointBuffer );
        model->m_pointCloud->setPointArray( points, numPoints );

        if (allColors.size())
        {
            model->m_pointCloud->setColorArray(pointColors, numPoints);
        }

        // Add sub cloud information
        if (sub_clouds.size())
        {
            indexArray sub_clouds_array = indexArray( new unsigned int[sub_clouds.size() * 2] );
            for(size_t i = 0; i < sub_clouds.size(); i++)
            {
                sub_clouds_array[i*2 + 0] = sub_clouds[i].first;
                sub_clouds_array[i*2 + 1] = sub_clouds[i].second;
            }

            model->m_pointCloud->addIndexChannel(sub_clouds_array, "sub_clouds", numPoints, 2);
        }
    }

}

void UosIO::readOldFormat(ModelPtr &model, string dir, int first, int last, size_t &n)
{
    Matrix4<Vec> m_tf;

    list<Vec > ptss;
    list<Vec > allPoints;
    for(int fileCounter = first; fileCounter <= last; fileCounter++)
    {
        float euler[6];
        ifstream scan_in, pose_in, frame_in;

        // Code imported from slam6d! Don't blame me..
        string scanFileName;
        string poseFileName;

        // Create correct path
        boost::filesystem::path p(
                boost::filesystem::path(dir) / 
                boost::filesystem::path( to_string( fileCounter, 3 ) ) /
                boost::filesystem::path( "position.dat" ) );

        // Get file name (if some knows a more elegant way to
        // extract the pull path let me know
        poseFileName = "/" + p.relative_path().string();

        // Try to open file
        pose_in.open(poseFileName.c_str());

        // Abort if opening failed and try with next die
        if (!pose_in.good()) continue;
        cout << timestamp << "Processing Scan " << dir << "/" << to_string(fileCounter, 3) << endl;

        // Extract pose information
        for (unsigned int i = 0; i < 6; pose_in >> euler[i++]);

        // Convert mm to cm
        for (unsigned int i = 0; i < 3; i++) euler[i] = euler[i] * 0.1;

        // Convert angles from deg to rad
        for (unsigned int i = 3; i <= 5; i++) {
            euler[i] *= 0.01f;
            //   if (euler[i] < 0.0) euler[i] += 360;
            euler[i] = rad(euler[i]);
        }

        // Read and convert scan
        for (int i = 1; ; i++) {
            //scanFileName = dir + to_string(fileCounter, 3) + "/scan" + to_string(i,3) + ".dat";

            boost::filesystem::path sfile(
                    boost::filesystem::path(dir) /
                    boost::filesystem::path( to_string( fileCounter, 3 ) ) /
                    boost::filesystem::path( "scan" + to_string(i) + ".dat" ) );
            scanFileName = "/" + sfile.relative_path().string();

            scan_in.open(scanFileName.c_str());
            if (!scan_in.good()) {
                scan_in.close();
                scan_in.clear();
                break;
            }


            int    Nr = 0, intensity_flag = 0;
            int    D;
            double current_angle;
            double X, Z, I;                     // x,z coordinate and intensity

            char firstLine[81];
            scan_in.getline(firstLine, 80);

            char cNr[4];
            cNr[0] = firstLine[2];
            cNr[1] = firstLine[3];
            cNr[2] = firstLine[4];
            cNr[3] = 0;
            Nr = atoi(cNr);

            // determine weather we have the new files with intensity information
            if (firstLine[16] != 'i') {
                intensity_flag = 1;
                char cAngle[8];
                cAngle[0] = firstLine[35];
                cAngle[1] = firstLine[36];
                cAngle[2] = firstLine[37];
                cAngle[3] = firstLine[38];
                cAngle[4] = firstLine[39];
                cAngle[5] = firstLine[40];
                cAngle[6] = firstLine[41];
                cAngle[7] = 0;
                current_angle = atof(cAngle);
                cout << current_angle << endl;
            } else {
                intensity_flag = 0;
                char cAngle[8];
                cAngle[0] = firstLine[54];
                cAngle[1] = firstLine[55];
                cAngle[2] = firstLine[56];
                cAngle[3] = firstLine[57];
                cAngle[4] = firstLine[58];
                cAngle[5] = firstLine[59];
                cAngle[6] = firstLine[60];
                cAngle[7] = 0;
                current_angle = atof(cAngle);
            }

            double cos_currentAngle = cos(rad(current_angle));
            double sin_currentAngle = sin(rad(current_angle));

            for (int j = 0; j < Nr; j++) {
                if (!intensity_flag) {
                    scan_in >> X >> Z >> D >> I;
                } else {
                    scan_in >> X >> Z;
                    I = 1.0;
                }

                // calculate 3D coordinates (local coordinates)
                Vec p;
                p[0] = X;
                p[1] = Z * sin_currentAngle;
                p[2] = Z * cos_currentAngle;

                ptss.push_back(p);
            }
            scan_in.close();
            scan_in.clear();
        }

        pose_in.close();
        pose_in.clear();

        // Create path to frame file
        boost::filesystem::path framePath(
                boost::filesystem::path(dir) / 
                boost::filesystem::path("scan" + to_string( fileCounter, 3 ) + ".frames" ) );
        string frameFileName = "/" + framePath.relative_path().string();

        // Try to open frame file
        frame_in.open(frameFileName.c_str());
        if(frame_in.good())
        {
            // Transform scan data according to frame file
            m_tf = parseFrameFile(frame_in);
        }
        else
        {
            // Transform scan data using information from 'position.dat'
            Vec position(euler[0], euler[1], euler[2]);
            Vec angle(euler[3], euler[4], euler[5]);
            m_tf = Matrix4<Vec>(position, angle);
        }

        // Transform points and insert in to global vector
        list<Vec >::iterator it;
        for(it = ptss.begin(); it != ptss.end(); it++)
        {
            Vec v = *it;
            v = m_tf * v;
            allPoints.push_back(v);
        }

        // Clear scan
        ptss.clear();
    }

    // Convert into indexed array
    if(allPoints.size() > 0)
    {
        cout << timestamp << "UOS Reader: Read " << allPoints.size() << " points." << endl;
        n = allPoints.size();
        floatArr points( new float[3 * allPoints.size()] );
        list<Vec >::iterator p_it;
        int i(0);
        for( p_it = allPoints.begin(); p_it != allPoints.end(); p_it++ )
        {
            int t_index = 3 * i;
            Vec v = *p_it;
            points[t_index    ] = v[0];
            points[t_index + 1] = v[1];
            points[t_index + 2] = v[2];
            i++;
        }

        // Alloc model
        model = ModelPtr( new Model );
        model->m_pointCloud = PointBufferPtr( new PointBuffer );
        model->m_pointCloud->setPointArray( points, n );
    }
}

Matrix4<Vec> UosIO::parseFrameFile(ifstream& frameFile)
{
    float m[16], color;
    while(frameFile.good())
    {
        for(int i = 0; i < 16; i++ && frameFile.good()) frameFile >> m[i];
        frameFile >> color;
    }

    return Matrix4<Vec>(m);
}

} // namespace lvr2
