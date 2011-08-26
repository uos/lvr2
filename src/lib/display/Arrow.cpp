/*
 * PointCloud.cpp
 *
 *  Created on: 02.09.2008
 *      Author: twiemann
 */

#include "Arrow.h"
#include "glut.h"


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
    rotation = Matrix4(alignxf);
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
    if(m_showAxes) glCallList(axesListIndex);

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


