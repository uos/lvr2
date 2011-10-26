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


 /*
 * PointCloud.cpp
 *
 *  Created on: 20.08.2011
 *      Author: Thomas Wiemann
 */

#include "PointCloud.hpp"
#include "ColorMap.hpp"

#include <string.h>

namespace lssr
{

PointCloud::PointCloud()
{

}

PointCloud::PointCloud(PointIO& loader, string name) : Renderable(name)
{
    int maxColors = 255;

    m_boundingBox = new BoundingBox<Vertex<float> >;
    m_pointLoader = &loader;

    size_t n_points;
    float** points = loader.getIndexedPointArray(n_points);
    uchar** colors = loader.getIndexedPointColorArray(n_points);
    float*  intensities = loader.getPointIntensityArray(n_points);

    ColorMap c_map(maxColors);

    for(size_t i = 0; i < loader.getNumPoints(); i++)
    {
        float x = points[i][0];
        float y = points[i][1];
        float z = points[i][2];

        unsigned char r, g, b;

        if(colors)
        {
            r = colors[i][0];
            g = colors[i][1];
            b = colors[i][2];
        }
        else if (intensities)
        {
            // Get intensity
            float color[3];
            c_map.getColor(color, (size_t)intensities[i], GREY);

            r = (uchar)(color[0] * 255);
            g = (uchar)(color[1] * 255);
            b = (uchar)(color[2] * 255);

        }
        else
        {
            r = 0;
            g = 255;
            b = 0;
        }

        m_boundingBox->expand(x, y, z);
        m_points.push_back(uColorVertex(x, y, z, r, g, b));
    }

    updateDisplayLists();
}

void PointCloud::updateDisplayLists(){

    // Check for existing display list for normal rendering
    if(m_listIndex != -1) {
        cout<<"PointCloud::initDisplayList() delete display list"<<endl;
        glDeleteLists(m_listIndex,1);
    }

    // Create new display list and render points
    m_listIndex = glGenLists(1);
    glNewList(m_listIndex, GL_COMPILE);
    glBegin(GL_POINTS);

    for(size_t i = 0; i < m_points.size(); i++)
    {
        float r = m_points[i].r / 255.0;
        float g = m_points[i].g / 255.0;
        float b = m_points[i].b / 255.0;

        glColor3f(r, g, b);
        glVertex3f(m_points[i].x,
                   m_points[i].y,
                   m_points[i].z);
    }
    glEnd();
    glEndList();

    // Check for existing list index for rendering a selected point
    // cloud

    if(m_activeListIndex != -1)
    {
        cout<<"PointCloud::initDisplayList() delete  active display list"<<endl;
        glDeleteLists(m_activeListIndex,1);
    }

    m_activeListIndex = glGenLists(1);
    glNewList(m_activeListIndex, GL_COMPILE);
    glBegin(GL_POINTS);

    glColor3f(1.0, 1.0, 0.0);
    for(size_t i = 0; i < m_points.size(); i++)
    {

        glVertex3f(m_points[i].x,
                   m_points[i].y,
                   m_points[i].z);
    }
    glEnd();
    glEndList();




}

PointCloud::~PointCloud() {
    // TODO Auto-generated destructor stub
}

}
