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

#include <lvr2/display/PointCloud.hpp>
#include <lvr2/display/ColorMap.hpp>

#include <string.h>

namespace lvr2
{

PointCloud::PointCloud()
{
    m_numNormals = 0;
    m_boundingBox = new BoundingBox<Vec>;
    m_renderMode = RenderPoints;
}

PointCloud::PointCloud( PointBufferPtr buffer, string name) : Renderable(name)
{
	m_model = ModelPtr(new Model(buffer));
	init(buffer);
}

PointCloud::PointCloud( ModelPtr model, string name) : Renderable(name)
{

    m_model = model;
    init(m_model->m_pointCloud);
}

void PointCloud::updateBuffer(PointBufferPtr buffer)
{
	init(buffer);

}

void PointCloud::init(PointBufferPtr buffer)
{
	int maxColors = 255;
	m_numNormals = 0;

	m_boundingBox = new BoundingBox<Vec>;
	m_renderMode = RenderPoints;

	if(buffer)
	{
		size_t n_points = buffer->numPoints();
		floatArr points = buffer->getPointArray();
        m_numNormals = 0;
		m_normals    = buffer->getNormalArray();

        if (m_normals)
            m_numNormals = n_points;

        unsigned w_color, dummy;
		ucharArr colors      = buffer->getColorArray(w_color);
		floatArr intensities = buffer->getFloatArray("intensities", n_points, dummy);

		ColorMap c_map(maxColors);

		for(size_t i = 0; i < n_points; i++)
		{
			float x = points[i*3 + 0];
			float y = points[i*3 + 1];
			float z = points[i*3 + 2];

			m_boundingBox->expand(Vector<Vec>(x,y,z));

			unsigned char r, g, b;

			if(colors)
			{
				r = colors[i*w_color + 0];
				g = colors[i*w_color + 1];
				b = colors[i*w_color + 2];
			}
			else if (intensities)
			{
				// Get intensity
				float color[3];
				c_map.getColor(color, (size_t)intensities[i], SHSV);

				r = (unsigned char)(color[0] * 255);
				g = (unsigned char)(color[1] * 255);
				b = (unsigned char)(color[2] * 255);

			}
			else
			{
				r = 0;
				g = 0;
				b = 0;
			}


			m_points.push_back(uColorVertex(x, y, z, r, g, b));
		}
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
        float r = m_points[i].r / 255.0f;
        float g = m_points[i].g / 255.0f;
        float b = m_points[i].b / 255.0f;

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

    float length = 0.01f * m_boundingBox->getRadius();

    // Create a new display list for normals
    if(m_numNormals)
    {
        m_normalListIndex = glGenLists(1);
        glNewList(m_normalListIndex, GL_COMPILE);
        glColor3f(1.0, 0.0, 1.0);
        for(int i = 0; i < m_numNormals; i++)
        {
            Vector<Vec> start(m_points[i].x, m_points[i].y, m_points[i].z);
            Normal<Vec> normal(m_normals[i*3 + 0], m_normals[i*3 + 1], m_normals[i*3 + 2]);
            Vector<Vec> end = start + normal * length;
            glBegin(GL_LINES);
            glVertex3f(start[0], start[1], start[2]);
            glVertex3f(end[0], end[1], end[2]);
            glEnd();
        }
        glEndList();
    }

}

PointCloud::~PointCloud() {
    // TODO Auto-generated destructor stub
}

} // namespace lvr2
