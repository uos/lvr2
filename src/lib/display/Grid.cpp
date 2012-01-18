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
 * Grid.cpp
 *
 *  @date 10.01.2012
 *  @author Thomas Wiemann
 */

#include "Grid.hpp"
namespace lssr
{

Grid::Grid(floatArr vertices, uintArr boxes, uint numPoints, uint numBoxes)
    : m_vertices(vertices), m_boxes(boxes), m_numPoints(numPoints), m_numBoxes(numBoxes)
{
    m_lineWidth = 2.0;
    m_pointSize = 5.0;

    // Create display lists for points and grid
    m_pointDisplayList = glGenLists(1);
    m_gridDisplayList = glGenLists(1);

    // Render points
    glNewList(m_pointDisplayList, GL_COMPILE);
    glBegin(GL_POINTS);
    for(int i = 0; i < m_numPoints; i++)
    {
        int pos = i * 4;
        float x = m_vertices[pos];
        float y = m_vertices[pos + 1];
        float z = m_vertices[pos + 2];
        float d = m_vertices[pos + 3];

        if(d > 0)
        {
            glColor3f(0.54, 0.17, 0.89);
        }
        else
        {
            glColor3f(1.0, 0.64, 0.0);
        }
        glVertex3f(x, y, z);

    }
    glEnd();
    glEndList();

    // Render boxes
    glNewList(m_gridDisplayList, GL_COMPILE);
    glColor3f(125.0, 125.0, 125.0);

    // Coordinates for box corners
    float x[8];
    float y[8];
    float z[8];

    // Get box corner coordinates
    for(int i = 0; i < m_numBoxes; i++)
    {
        int box_pos = i * 8;
        for(int j = 0; j < 8; j++)
        {
            int vertex_pos = 4 * m_boxes[box_pos + j];
            x[j] = m_vertices[vertex_pos];
            y[j] = m_vertices[vertex_pos + 1];
            z[j] = m_vertices[vertex_pos + 2];

            m_boundingBox->expand(x[j], y[j], z[j]);
        }

        // Render quads
        glBegin(GL_LINE_LOOP);
        glVertex3f(x[0], y[0], z[0]);
        glVertex3f(x[1], y[1], z[1]);
        glVertex3f(x[2], y[2], z[2]);
        glVertex3f(x[3], y[3], z[3]);
        glEnd();

        glBegin(GL_LINE_LOOP);
        glVertex3f(x[4], y[4], z[4]);
        glVertex3f(x[5], y[5], z[5]);
        glVertex3f(x[6], y[6], z[6]);
        glVertex3f(x[7], y[7], z[7]);
        glEnd();

        glBegin(GL_LINES);
        glVertex3f(x[0], y[0], z[0]);
        glVertex3f(x[4], y[4], z[4]);

        glVertex3f(x[1], y[1], z[1]);
        glVertex3f(x[5], y[5], z[5]);

        glVertex3f(x[3], y[3], z[3]);
        glVertex3f(x[7], y[7], z[7]);

        glVertex3f(x[2], y[2], z[2]);
        glVertex3f(x[6], y[6], z[6]);
        glEnd();
    }

    glEndList();

}

void Grid::render()
{
    glLineWidth(m_lineWidth);
    glCallList(m_gridDisplayList);
    glPointSize(m_pointSize);
    glCallList(m_pointDisplayList);
    glPointSize(1.0);
    glLineWidth(1.0);
}

Grid::~Grid()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lssr */
