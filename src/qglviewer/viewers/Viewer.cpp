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
 * Viewer.cpp
 *
 *  Created on: 15.09.2010
 *      Author: Thomas Wiemann
 */

#include "Viewer.h"

Viewer::Viewer(QWidget* parent, const QGLWidget* shared)
	: QGLViewer(parent, shared),  m_parent(parent), m_zoom(1.0)
{
    m_kfi = new qglviewer::KeyFrameInterpolator(new qglviewer::Frame());


    connect(m_kfi, SIGNAL(interpolated()), this, SLOT(updateGL()));
    connect(m_kfi, SIGNAL(interpolated()), this, SLOT(createSnapshot()));

}


Viewer::~Viewer()
{
	// TODO Auto-generated destructor stub
}

void Viewer::draw()
{
	setTextIsEnabled(true);
	float pos[3];

    if(m_kfi->interpolationIsStarted())
    {
        camera()->setPosition(m_kfi->frame()->position());
        camera()->setOrientation(m_kfi->frame()->orientation());
    }

    glScalef(m_zoom, m_zoom, m_zoom);
    list<Visualizer*>::iterator it;
    for(it = m_dataObjects.begin(); it != m_dataObjects.end(); it++)
    {
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        (*it)->renderable()->render();
        glPopAttrib();
    }
}

void Viewer::createSnapshot()
{
    saveSnapshot(true);
}

void Viewer::resetCamera()
{
	qglviewer::Vec center(0, 0, 0);
	setSceneCenter(center);

	Vertex<float> v_min = m_boundingBox.getMin();
	Vertex<float> v_max = m_boundingBox.getMax();

	qglviewer::Vec v1(v_min.x, v_min.y, v_min.z);
	qglviewer::Vec v2(v_max.x, v_max.y, v_max.z);

	cout << v_min;
	cout << v_max;

	setSceneBoundingBox(v1, v2);
	m_zoom = 1.0;

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
    Vertex<float> v_min = bb->getMin();
    Vertex<float> v_max = bb->getMax();

    qglviewer::Vec v1(v_min.x, v_min.y, v_min.z);
    qglviewer::Vec v2(v_max.x, v_max.y, v_max.z);

    setSceneBoundingBox(v1, v2);

    m_near = camera()->zNear();
    m_far  = camera()->zFar();
    m_zoom = 1.0;
    showEntireScene();
    updateGL();
}

void Viewer::addDataObject(Visualizer* obj)
{
	if(obj->renderable())
	{
		BoundingBox<Vertex<float> >* bb = (obj->renderable()->boundingBox());
		if(bb->isValid())
		{
			m_boundingBox.expand(*bb);
			resetCamera();
		}
		m_dataObjects.push_back(obj);
	}
}

void Viewer::removeDataObject(Visualizer* obj)
{
    m_dataObjects.remove(obj);
}

void Viewer::removeDataObject(CustomTreeWidgetItem* item)
{
    list<Visualizer*>::iterator it = m_dataObjects.begin();

    while(it != m_dataObjects.end())
    {
        Visualizer* d = *it;
        if(d->renderable() == item->renderable())
        {
            break;
        }
        it++;
    }
    m_dataObjects.erase(it);
    updateGL();
}

void Viewer::updateDataObject(Visualizer* obj)
{
	updateGL();
}
