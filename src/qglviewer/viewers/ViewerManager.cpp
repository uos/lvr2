/*
 * ViewerManager.cpp
 *
 *  Created on: 07.10.2010
 *      Author: Thomas Wiemann
 */

#include "ViewerManager.h"
#include "PerspectiveViewer.h"


ViewerManager::ViewerManager(QWidget* parent, const QGLWidget* shared)
{
	m_currentViewer = new PerspectiveViewer(parent, shared);
	m_parentWidget = parent;
}

ViewerManager::~ViewerManager()
{

}

Viewer* ViewerManager::current()
{
	return m_currentViewer;
}

void ViewerManager::addDataCollector(DataCollector* c)
{
    cout << "ADD DATA COLLECTOR" << endl;
	// Stub, currently support only one single viewer instance
	m_currentViewer->addDataObject(c);
}

void ViewerManager::updateDataObject(DataCollector* obj)
{
	m_currentViewer->updateDataObject(obj);
}
