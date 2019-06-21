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
 * Grid.cpp
 *
 *  @date 10.01.2012
 *  @author Thomas Wiemann
 */

#include "lvr2/display/Grid.hpp"

namespace lvr2
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
            glColor3f(0.54f, 0.17f, 0.89f);
        }
        else
        {
            glColor3f(1.0f, 0.64f, 0.0f);
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
    for(unsigned int i = 0; i < m_numBoxes; i++)
    {
        int box_pos = i * 8;
        for(int j = 0; j < 8; j++)
        {
            int vertex_pos = 4 * m_boxes[box_pos + j];
            x[j] = m_vertices[vertex_pos];
            y[j] = m_vertices[vertex_pos + 1];
            z[j] = m_vertices[vertex_pos + 2];

            m_boundingBox->expand(Vec(x[j], y[j], z[j]));
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
	if(m_active)
	{
		glLineWidth(m_lineWidth);
		glCallList(m_gridDisplayList);
		glPointSize(m_pointSize);
		glCallList(m_pointDisplayList);
		glPointSize(1.0);
		glLineWidth(1.0);
	}
}

Grid::~Grid()
{
    // TODO Auto-generated destructor stub
}

} /* namespace lvr2 */
