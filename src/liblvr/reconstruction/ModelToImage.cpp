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
        int minZ, int maxZ,
        int minHorizontenAngle, int maxHorizontalAngle,
        int mainVerticalAngle, int maxVerticalAngle,
        bool imageOptimization,
        bool leftHandedInputData)
{
    m_width = width;
    m_height = height;
    m_leftHanded = leftHandedInputData;


    int min_range = + 1e7;
    int max_range = - 1e7;

    Projection* p = new EquirectangularProjection(14800, 1000, -360, 360, -90, 120, true);

    int** img = new int*[p->h()];
    for(int i = 0; i < p->h(); i++)
    {
        img[i] = new int[p->w()];
    }

    for(int i = 0; i < p->h(); i++)
    {
        for(int j = 0; j < p->w(); j++)
        {
            img[i][j] = 0;
        }
    }

    size_t n_points;
    floatArr points = buffer->getPointArray(n_points);

    std::list<std::tuple<int, int, int> >pixels;

    int img_x, img_y, range;
    for(int i = 0; i < n_points; i++)
    {
        p->project(img_x, img_y, range, points[3 * i], points[3 * i + 1], points[3 * i + 2]);
        if(range > max_range)
        {
            max_range = range;
        }

        if(range < min_range)
        {
            min_range = range;
        }
        //cout << img_x << " " << img_y << " " << range << endl;
        pixels.push_back(std::tuple<int, int, int>(img_x, img_y, range));
    }

    if(max_range > 3000) max_range = 3000;

    int interval = max_range - min_range;
    //cout << max_range << " " << min_range << " " << interval << endl;

    for(auto it : pixels)
    {
        int i = std::get<0>(it);
        int j = std::get<1>(it);
        int r = std::get<2>(it);

        if(r > max_range) r = max_range;

        int val = (float)(r - min_range) / interval * 255;
        //cout << i << " " << j <<  " " << val << endl;
        img[j][i] = val;
    }

    std::ofstream out("img.pgm");
    out << "P2" << endl;
    out << p->w() << " " << p->h() << " 255" << endl;
    for(int i = 0; i < p->h(); i++)
    {
        for(int j = 0; j < p->w(); j++)
        {
            out << img[i][j] << " ";
        }
    }
    out.close();
    delete p;
}



ModelToImage::~ModelToImage()
{
	// TODO Auto-generated destructor stub
}

} /* namespace lvr */
