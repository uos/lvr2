/* Copyright (C) 2011 Uni Osnabrück
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

/**
 * BoctreeIO.cpp
 *
 *  @date 23.08.2012
 *  @author Thomas Wiemann
 */

#include <stdio.h>

#include "io/BoctreeIO.hpp"
#include "io/Timestamp.hpp"

#include "geometry/Matrix4.hpp"

#include "slam6d/Boctree.h"

namespace lvr
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
        if(string(p.extension().c_str()) == ".oct")
        {
            // Check for naming convention "scanxxx.3d"
            int num = 0;
            if(sscanf(p.filename().c_str(), "scan%3d", &num))
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

            Matrix4<float> tf;
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
                    Vertex<float> position(euler[0], euler[1], euler[2]);
                    Vertex<float> angle(euler[3], euler[4], euler[5]);
                    tf = Matrix4<float>(position, angle);
                }
                else
                {
                    cout << timestamp << "UOS Reader: Warning: No position information found." << endl;
                    tf = Matrix4<float>();
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
                Vertex<float> v(points[j].x, points[j].y, points[j].z);
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
        ucharArr colors(new uchar[3 * allPoints.size()]);
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
        model->m_pointCloud->setPointColorArray(colors, allPoints.size());
        model->m_pointCloud->setPointIntensityArray(intensities, allPoints.size());
    }

    return model;
}

void BoctreeIO::save( string filename )
{

}

Matrix4<float>  BoctreeIO::parseFrameFile(ifstream& frameFile)
{
    float m[16], color;
    while(frameFile.good())
    {
        for(int i = 0; i < 16; i++ && frameFile.good()) frameFile >> m[i];
        frameFile >> color;
    }

    return Matrix4<float>(m);
}

} /* namespace lvr */
