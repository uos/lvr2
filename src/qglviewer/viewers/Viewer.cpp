/*
 * Viewer.cpp
 *
 *  Created on: 15.09.2010
 *      Author: Thomas Wiemann
 */

#include "Viewer.h"

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

	Vertex<float> v_min = m_boundingBox.getMin();
	Vertex<float> v_max = m_boundingBox.getMax();

	qglviewer::Vec v1(v_min.x, v_min.y, v_min.z);
	qglviewer::Vec v2(v_max.x, v_max.y, v_max.z);

	setSceneBoundingBox(v1, v2);
	showEntireScene();
}

void Viewer::centerViewOnObject(Renderable* renderable)
{
    BoundingBox<Vertex<float> >* bb = renderable->boundingBox();

    // Center view on selected object
    Vertex<float> centroid = bb->getCentroid();
    qglviewer::Vec center(centroid[0], centroid[1], centroid[2]);
    setSceneCenter(center);

    // Set new scene boundaries
    Vertex<float> v_min = m_boundingBox.getMin();
    Vertex<float> v_max = m_boundingBox.getMax();

    qglviewer::Vec v1(v_min.x, v_min.y, v_min.z);
    qglviewer::Vec v2(v_max.x, v_max.y, v_max.z);

    setSceneBoundingBox(v1, v2);
    showEntireScene();
}

void Viewer::addDataObject(DataCollector* obj)
{
	BoundingBox<Vertex<float> >* bb = (obj->renderable()->boundingBox());
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
