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
 * PerspectiveViewer.cpp
 *
 *  Created on: 22.09.2010
 *      Author: Thomas Wiemann
 */

#include "PerspectiveViewer.h"
#include "OrthoCamera.h"

#include <iostream>
using std::cout;
using std::endl;

#include "geometry/Vertex.hpp"

ViewerType PerspectiveViewer::type()
{
	return PERSPECTIVE_VIEWER;
}

PerspectiveViewer::PerspectiveViewer(QWidget* parent, const QGLWidget* shared)
	: Viewer(parent, shared)
{

	// Initialize the camera array
	for(int i = 0; i < 4; i++) m_camera[i] = 0;

	// Save pointer to the current (perspective) camera
	m_camera[PERSPECTIVE] = camera();

	// Set standard projection mode to perspective. The other
	// camera pointers are instantiated only when needed
	m_projectionMode = PERSPECTIVE;
	m_showFog = false;
	m_fogType = FOG_LINEAR;

	// Set a custom mouse binding for look around mod
}

PerspectiveViewer::~PerspectiveViewer()
{
	// TODO Auto-generated destructor stub
}

void PerspectiveViewer::init()
{
	// Light parameters
	float LightPos[4]={-5.0f,5.0f,10.0f,0.0f};
	float Ambient[4]={0.5f,0.5f,0.5f,1.0f};
	float FogCol[3]={0.8f,0.8f,0.8f};  // define a nice light grey

	 // Turn on Lighting
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0,GL_POSITION,LightPos);
	glLightfv(GL_LIGHT0,GL_AMBIENT,Ambient);

	// Enable fog
	glFogfv(GL_FOG_COLOR,FogCol); // Set the fog color
	glFogf(GL_FOG_DENSITY, 1.0f);  // Thin the fog out a little
	glFogi(GL_FOG_MODE, GL_LINEAR);

	glEnable(GL_DEPTH_TEST);

	createBackgroundDisplayList();

	setMouseBinding(Qt::SHIFT + Qt::LeftButton, CAMERA, LOOK_AROUND  );
}

void PerspectiveViewer::toggleFog()
{
	m_showFog = !m_showFog;
	if(m_showFog)
	{
		glEnable(GL_FOG);
	}
	else
	{
		glDisable(GL_FOG);
	}
}

void PerspectiveViewer::setFogDensity(float density)
{
	if(m_showFog){
		glFogf(GL_FOG_DENSITY, density);
		activateWindow();
	}
}

void PerspectiveViewer::setFogType(FOGTYPE f)
{
	switch(f)
	{
	case FOG_LINEAR: 	glFogi(GL_FOG_MODE, GL_LINEAR);	break;
	case FOG_EXP:		glFogi(GL_FOG_MODE, GL_EXP); 	break;
	case FOG_EXP2:		glFogi(GL_FOG_MODE, GL_EXP2); 	break;
	}
}

void PerspectiveViewer::createBackgroundDisplayList()
{
	m_backgroundDisplayList = glGenLists(1);
	glNewList(m_backgroundDisplayList, GL_COMPILE);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glDisable(GL_LIGHTING);

	glBegin(GL_QUADS);

	//light blue
	glColor3f(0.80, 0.80, 0.80);
	glVertex2f(-1.0, 1.0);
	glVertex2f(1.0, 1.0);

	//blue color
	glColor3f(0.725, 0.827, 0.933);
	glVertex2f(1.0, -1.0);
	glVertex2f(-1.0, -1.0);
	glEnd();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glEndList();
}

void PerspectiveViewer::draw()
{
//	// Render background
	glDisable(GL_DEPTH_TEST);
	glCallList(m_backgroundDisplayList);
	glEnable(GL_DEPTH_TEST);

	// Draw contends
	Viewer::draw();

}

void PerspectiveViewer::setProjectionMode(ProjectionMode mode)
{
	m_projectionMode = mode;

	// Create and setup a new cam if necessary
	if(m_camera[mode] == 0)
	{
		m_camera[mode] = new OrthoCamera;
		switch(mode)
		{
			case ORTHOXZ: m_camera[mode]->setPosition(qglviewer::Vec(0.0, 1.0, 0.0)); break;
			case ORTHOXY: m_camera[mode]->setPosition(qglviewer::Vec(0.0, 0.0, 1.0)); break;
			case ORTHOYZ: m_camera[mode]->setPosition(qglviewer::Vec(1.0, 0.0, 0.0)); break;
			case PERSPECTIVE: break; // Should never get here
		}

		// Setup scene boundary
		Vertex<float> v_min = m_boundingBox.getMin();
		Vertex<float> v_max = m_boundingBox.getMax();
		m_camera[mode]->setSceneBoundingBox(qglviewer::Vec(v_min.x, v_min.y, v_min.z),
											qglviewer::Vec(v_max.x, v_max.y, v_max.z));

		// Set cam to look at entire scene
		m_camera[mode]->lookAt(sceneCenter());
		m_camera[mode]->setType(qglviewer::Camera::ORTHOGRAPHIC);
		m_camera[mode]->showEntireScene();

		// Setup constraints
		qglviewer::WorldConstraint* constraint = new qglviewer::WorldConstraint();
		constraint->setRotationConstraintType(qglviewer::AxisPlaneConstraint::FORBIDDEN);
		m_camera[mode]->frame()->setConstraint(constraint);
	}

	// Switch camera
	setCamera(m_camera[mode]);
}


