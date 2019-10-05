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
 * PointCloud.cpp
 *
 *  Created on: 02.09.2008
 *      Author: twiemann
 */

#include "lvr2/display/Arrow.hpp"
#include <GL/glut.h>

namespace lvr2
{

Arrow::Arrow(string filename) : Renderable(filename){
    color = 0;
}

Arrow::Arrow(int color){
    this->color = color;
}

void Arrow::setPosition(double x, double y, double z, double roll, double pitch, double yaw) {
    this->m_position.x = x;
    this->m_position.y = y;
    this->m_position.z = z;
    this->roll = roll;
    this->pitch = pitch;
    this->yaw = yaw;

    double rot[3];
    double pos[3];
    rot[0] =  roll; //pitch; //x-axis
    rot[1] =  pitch; //y-axis
    rot[2] =  yaw; //z-axis
    pos[0] =  m_position.x;
    pos[1] =  m_position.y;
    pos[2] =  m_position.z;

    //	Quat quat(rot, pos);

    float alignxf[16];
    EulerToMatrix(pos, rot, alignxf);

    //	quat.getMatrix(alignxf);
    rotation = Matrix4<BaseVector<float>>(alignxf);
}

Arrow::~Arrow() {
    // TODO Auto-generated destructor stub
}

void Arrow::render() {
    //	float radius = 30.0f;
    //	float length = 150.0f;
    //	int nbSubdivisions = 4;

    glPushMatrix();
    glMultMatrixf(m_transformation.getData());
    glMultMatrixf(rotation.getData());
    if(m_showAxes) glCallList(m_axesListIndex);

    glDisable(GL_LIGHTING);
    glDisable(GL_BLEND);

    switch (color) {
    case 0: glColor4f(1.0f, 0.0f, 0.0f, 1.0f); break;
    case 1: glColor4f(0.0f, 1.0f, 0.0f, 1.0f); break;
    default: glColor4f(0.0f, 0.0f, 1.0f, 1.0f); break;
    }

    glRotated(90.0, 0.0, 1.0, 0.0);

    GLUquadric* quadric = gluNewQuadric();
    //	gluCylinder (quadric, radius, 0, length, nbSubdivisions, 1 );
    //
    // 	glBindTexture(GL_TEXTURE_2D, theTexture);
    //
    //  glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);

    float xLength = 50;
    float yLength = 30;
    float zLength = 80;

    glColor4f(0.0f, 1.0f, 1.0f, 1.0f);
    glVertex3f( xLength, yLength,-zLength);
    glVertex3f(-xLength, yLength,-zLength);
    glVertex3f(-xLength, yLength, zLength);
    glVertex3f( xLength, yLength, zLength);

    glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
    glVertex3f( xLength,-yLength, zLength);
    glVertex3f(-xLength,-yLength, zLength);
    glVertex3f(-xLength,-yLength,-zLength);
    glVertex3f( xLength,-yLength,-zLength);

    // front and back
    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
    glVertex3f(xLength, yLength, zLength);
    glVertex3f(-xLength, yLength, zLength);
    glVertex3f(-xLength,-yLength, zLength);
    glVertex3f( xLength,-yLength, zLength);

    glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
    glVertex3f( xLength,-yLength,-zLength);
    glVertex3f(-xLength,-yLength,-zLength);
    glVertex3f(-xLength, yLength,-zLength);
    glVertex3f( xLength, yLength,-zLength);

    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
    glVertex3f(-xLength, yLength, zLength);
    glVertex3f(-xLength, yLength,-zLength);
    glVertex3f(-xLength,-yLength,-zLength);
    glVertex3f(-xLength,-yLength, zLength);

    glVertex3f( xLength, yLength,-zLength);
    glVertex3f( xLength, yLength, zLength);
    glVertex3f( xLength,-yLength, zLength);
    glVertex3f( xLength,-yLength,-zLength);

    glEnd();
    // 	glDisable(GL_TEXTURE_2D);


    glDisable(GL_LIGHTING);
    gluDeleteQuadric ( quadric );
    glPopMatrix();
    glPopAttrib();
    glEnable(GL_BLEND);
}

} // namespace lvr2
