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

#include "lvr2/display/PointCorrespondences.hpp"

namespace lvr2
{

PointCorrespondences::PointCorrespondences(std::vector<float> points1, std::vector<float> points2)
{
    m_displayList = glGenLists(1);
    glNewList(m_displayList, GL_COMPILE);

    glBegin(GL_POINTS);
    // Render points
    for(size_t i = 0; i < points1.size() / 3; i++)
    {
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(points1[i * 3], points1[i * 3 + 1], points1[i * 3 + 2]);
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(points2[i * 3], points2[i * 3 + 1], points2[i * 3 + 2]);
    }
    glEnd();
    for(size_t i = 0; i < points1.size() / 3; i++)
    {
        glBegin(GL_LINES);
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(points1[i * 3], points1[i * 3 + 1], points1[i * 3 + 2]);
        glVertex3f(points2[i * 3], points2[i * 3 + 1], points2[i * 3 + 2]);
        glEnd();
    }
    glEndList();


}

void PointCorrespondences::PointCorrespondences::render()
{
    glLineWidth(1.0);
    glPointSize(5.0);
    glCallList(m_displayList);
}

} // namespace lvr2
