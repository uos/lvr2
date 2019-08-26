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

/*
 * ModelToImage.cpp
 *
 *  Created on: Jan 25, 2017
 *      Author: Thomas Wiemann (twiemann@uos.de)
 */

#include "lvr2/reconstruction/ModelToImage.hpp"
#include "lvr2/reconstruction/Projection.hpp"
#include "lvr2/io/Progress.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/geometry/BaseVector.hpp"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <list>
using namespace std;

namespace lvr2
{

using Vec = BaseVector<float>;

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

void ModelToImage::computeDepthListMatrix(DepthListMatrix& mat)
{
    cout << timestamp << "Initializting DepthListMatrix with dimensions " << m_width << " x " << m_height << endl;
    // Set correct image width and height
    for(int i = 0; i < m_height; i++)
    {
        mat.pixels.emplace_back(vector<vector<PanoramaPoint> >());
        for(int j = 0; j < m_width; j++)
        {
            mat.pixels[i].push_back(vector<PanoramaPoint>());
        }
    }

    // Get point array and size from buffer
    size_t n_points = m_points->numPoints();
    floatArr points = m_points->getPointArray();

    // Create progress output
    string comment = timestamp.getElapsedTime() + "Projecting points ";
    ProgressBar progress(n_points, comment);

    float range;
    int img_x, img_y;
    for(int i = 0; i < n_points; i++)
    {        
        Vec ppt(points[3 * i], points[3 * i + 1], points[3 * i + 2]);

        m_projection->project(
                    img_x, img_y, range,
                    ppt.x, ppt.y, ppt.z);

        // Update min and max ranges
        if(range > mat.maxRange)
        {
            mat.maxRange = range;
        }

        if(range < mat.minRange)
        {
            mat.minRange = range;
        }

        if(range < m_maxZ)
        {
            // Add point index to image pixel
            mat.pixels[img_y][img_x].emplace_back(PanoramaPoint(i));
        }
        ++progress;
    }
    cout << endl;
}

void ModelToImage::computeDepthImage(ModelToImage::DepthImage& img, ModelToImage::ProjectionPolicy policy)
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
    size_t n_points = m_points->numPoints();
    floatArr points = m_points->getPointArray();

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

void ModelToImage::writePGM(string filename, float cutoff)
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
