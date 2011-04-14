/*
 * Viewer.cpp
 *
 *  Created on: 15.09.2010
 *      Author: Thomas Wiemann
 */

#include "Viewer.h"

#include "../data/DataCollector.h"

#include <iostream>

#include <lib3d/PointCloud.h>

Viewer::Viewer(QWidget* parent, const QGLWidget* shared)
	: QGLViewer(parent, shared),  m_parent(parent) {}


Viewer::~Viewer()
{
	// TODO Auto-generated destructor stub
}

void Viewer::draw()
{
	list<DataCollector*>::iterator it;
	for(it = m_dataObjects.begin(); it != m_dataObjects.end(); it++)
	{
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		(*it)->renderable()->render();
		glPopAttrib();
	}
}

void Viewer::resetCamera()
{
	qglviewer::Vec center(0, 0, 0);
	setSceneCenter(center);
	qglviewer::Vec v1(m_boundingBox.v_min.x, m_boundingBox.v_min.y, m_boundingBox.v_min.z);
	qglviewer::Vec v2(m_boundingBox.v_max.x, m_boundingBox.v_max.y, m_boundingBox.v_max.z);
	setSceneBoundingBox(v1, v2);
	showEntireScene();
}

void Viewer::addDataObject(DataCollector* obj)
{
	BoundingBox* bb = (obj->renderable()->boundingBox());
	if(bb->isValid())
	{
		m_boundingBox.expand(*bb);
		resetCamera();
	}
	m_dataObjects.push_back(obj);
}

void Viewer::updateDataObject(DataCollector* obj)
{
	updateGL();
}
