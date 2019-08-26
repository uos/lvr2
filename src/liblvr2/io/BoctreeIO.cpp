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
 * BoctreeIO.cpp
 *
 *  @date 23.08.2012
 *  @author Thomas Wiemann
 */

#include <stdio.h>

#include "lvr2/io/BoctreeIO.hpp"
#include "lvr2/io/Timestamp.hpp"

#include <slam6d/Boctree.h>

namespace lvr2
{

BoctreeIO::BoctreeIO()
{

}

BoctreeIO::~BoctreeIO()
{

}

ModelPtr BoctreeIO::read(string directory )
{
    int numScans = 0;
    int firstScan = -1;
    int lastScan = -1;

    // First, look for .oct files
    boost::filesystem::directory_iterator lastFile;
    for(boost::filesystem::directory_iterator it(directory); it != lastFile; it++ )
    {
        boost::filesystem::path p = it->path();
        if(string(p.extension().string().c_str()) == ".oct")
        {
            // Check for naming convention "scanxxx.3d"
            int num = 0;
            if(sscanf(p.filename().string().c_str(), "scan%3d", &num))
            {
                numScans++;
                if(firstScan == -1) firstScan = num;
                if(lastScan == -1) lastScan = num;

                if(num > lastScan) lastScan = num;
                if(num < firstScan) firstScan = num;
            }
        }
    }




    list<Point> allPoints;
    if(numScans)
    {

        for(int i = 0; i < numScans; i++)
        {
            char scanfile[128];
            sprintf(scanfile, "%sscan%03d.oct", directory.c_str(), firstScan + i);

            cout << timestamp << "Reading " << scanfile << endl;

            Matrix4<Vec> tf;
            vector<Point> points;
            BOctTree<float>::deserialize(scanfile, points);

            // Try to get transformation from .frames file
            boost::filesystem::path frame_path(
                    boost::filesystem::path(directory) /
                    boost::filesystem::path( "scan" + to_string( firstScan + i, 3 ) + ".frames" ) );
            string frameFileName = "/" + frame_path.relative_path().string();

            ifstream frame_in(frameFileName.c_str());
            if(!frame_in.good())
            {
                // Try to parse .pose file
                boost::filesystem::path pose_path(
                        boost::filesystem::path(directory) /
                        boost::filesystem::path( "scan" + to_string( firstScan + i, 3 ) + ".pose" ) );
                string poseFileName = "/" + pose_path.relative_path().string();

                ifstream pose_in(poseFileName.c_str());
                if(pose_in.good())
                {
                    float euler[6];
                    for(int i = 0; i < 6; i++) pose_in >> euler[i];
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

            // Ugly hack to transform scan data
            for(int j = 0; j < points.size(); j++)
            {
                Vec v(points[j].x, points[j].y, points[j].z);
                v = tf * v;
                if(v.length() < 10000)
                {
                    points[j].x = v[0];
                    points[j].y = v[1];
                    points[j].z = v[2];
                    allPoints.push_back(points[j]);
                }

            }


        }
    }

    cout << timestamp << "Read " << allPoints.size() << " points." << endl;

    ModelPtr model( new Model );

    if(allPoints.size())
    {
        floatArr points(new float[3 * allPoints.size()]);
        ucharArr colors(new unsigned char[3 * allPoints.size()]);
        floatArr intensities( new float[allPoints.size()]);

        int i = 0;
        bool found_color = false;
        float min_r = 1e10;
        float max_r = -1e10;
        for(list<Point>::iterator it = allPoints.begin(); it != allPoints.end(); it++)
        {
            Point p = *it;

            points[3 * i    ] = p.x;
            points[3 * i + 1] = p.y;
            points[3 * i + 2] = p.z;

            colors[3 * i    ] = p.rgb[0];
            colors[3 * i + 1] = p.rgb[1];
            colors[3 * i + 2] = p.rgb[2];

            if(p.rgb[0] != 255 && p.rgb[1] != 255 && p.rgb[2] != 255)
            {
                found_color = true;
            }

            intensities[i] = p.reflectance;
            if(intensities[i] < min_r) min_r = intensities[i];
            if(intensities[i] > max_r) max_r = intensities[i];
            //cout << p.reflectance << " " << p.amplitude << " " << p.deviation << endl;
            i++;
        }

//        // Map reflectances to 0..255
//        float r_diff = max_r - min_r;
//        if(r_diff > 0)
//        {
//            size_t np = allPoints.size();
//            float b_size = r_diff / 255.0;
//            for(int a = 0; a < np; a++)
//            {
//                float value = intensities[a];
//                value -= min_r;
//                value /= b_size;
//                //cout << value << endl;
//                intensities[a] = value;
//            }
//        }

        model->m_pointCloud = PointBufferPtr( new PointBuffer );
        model->m_pointCloud->setPointArray(points, allPoints.size());
        model->m_pointCloud->setColorArray(colors, allPoints.size());
        model->m_pointCloud->addFloatChannel(intensities, "intensities", allPoints.size(), 1);
    }

    return model;
}

void BoctreeIO::save( string filename )
{

}

Matrix4<Vec>  BoctreeIO::parseFrameFile(ifstream& frameFile)
{
    float m[16], color;
    while(frameFile.good())
    {
        for(int i = 0; i < 16; i++ && frameFile.good()) frameFile >> m[i];
        frameFile >> color;
    }

    return Matrix4<Vec>(m);
}

} /* namespace lvr2 */
