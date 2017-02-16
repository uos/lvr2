/* Copyright (C) 2016 Uni Osnabrück
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

/*
 * ModelToImage.cpp
 *
 *  Created on: Jan 25, 2017
 *      Author: Thomas Wiemann (twiemann@uos.de)
 */

#include <lvr/reconstruction/ModelToImage.hpp>
#include <lvr/reconstruction/Projection.hpp>
#include <lvr/io/Progress.hpp>
#include <lvr/io/Timestamp.hpp>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <list>
using namespace std;

namespace lvr {


ModelToImage::ModelToImage(
        PointBufferPtr buffer,
        ModelToImage::ProjectionType projection,
        int width, int height,
        float minZ, float maxZ,
        int minHorizontenAngle, int maxHorizontalAngle,
        int minVerticalAngle, int maxVerticalAngle,
        bool imageOptimization,
        CoordinateSystem system)
{

    // Initialize global members
    m_width = width;
    m_height = height;
    m_coordinateSystem = system;
    m_minZ = minZ;
    m_maxZ = maxZ;
    m_minHAngle = minHorizontenAngle;
    m_maxHAngle = maxHorizontalAngle;
    m_minVAngle = minVerticalAngle;
    m_maxVAngle = maxVerticalAngle;
    m_points = buffer;

    // Create the projection representation
    m_projection = new EquirectangularProjection(
                m_width, m_height,
                minHorizontenAngle, maxHorizontalAngle,
                minVerticalAngle, maxVerticalAngle,
                imageOptimization, system);

}



ModelToImage::~ModelToImage()
{
    // TODO Auto-generated destructor stub
}

void lvr::ModelToImage::computeDepthImage(lvr::ModelToImage::DepthImage& img, lvr::ModelToImage::ProjectionPolicy policy)
{
    cout << timestamp << "Computing depth image. Image dimensions: " << m_width << " x " << m_height << endl;

    // Set correct image width and height
    for(int i = 0; i < m_height; i++)
    {
        img.pixels.emplace_back(vector<float>());
        for(int j = 0; j < m_width; j++)
        {
            img.pixels[i].push_back(0.0f);
        }
    }


    // Get point array and size from buffer
    size_t n_points;
    floatArr points = m_points->getPointArray(n_points);

    // Create progress output
    string comment = timestamp.getElapsedTime() + "Projecting points ";
    ProgressBar progress(n_points, comment);

    float range;
    int img_x, img_y;
    for(int i = 0; i < n_points; i++)
    {
        m_projection->project(
                    img_x, img_y, range,
                    points[3 * i], points[3 * i + 1], points[3 * i + 2]);

        // Update min and max ranges
        if(range > img.maxRange)
        {
            img.maxRange = range;
        }

        if(range < img.minRange)
        {
            img.minRange = range;
        }

        img.pixels[img_y][img_x] = range;
        ++progress;
    }
    cout << endl;
    cout << timestamp << "Min / Max range: " << img.minRange << " / " << img.maxRange << endl;
}

void lvr::ModelToImage::writePGM(string filename, float cutoff)
{
    // Compute panorama image
    ModelToImage::DepthImage img;
    computeDepthImage(img);

    // Clamp range values
    float min_r = std::min(m_minZ, img.minRange);
    float max_r = std::min(m_maxZ, img.maxRange);
    float interval = max_r - min_r;

    cout << min_r << " " << max_r << " " << interval << endl;

    // Open file, write header and pixel values
    std::ofstream out(filename);
    out << "P2" << endl;
    out << img.pixels[0].size() << " " << img.pixels.size() << " 255" << endl;

    for(int i = 0; i < img.pixels.size(); i++)
    {
        for(int j = 0; j < img.pixels[i].size(); j++)
        {
            int val = img.pixels[i][j];

            // Image was initialized with zeros. Fix that to
            // the measured min value
            if(val < min_r)
            {
                val = min_r;
            }

            val = (int)((float)(val - min_r) / interval * 255);
            out << val << " ";
        }
    }
}

} /* namespace lvr */
