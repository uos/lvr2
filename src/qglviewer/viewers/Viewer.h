/*
 * Viewer.h
 *
 *  Created on: 15.09.2010
 *      Author: Thomas Wiemann
 */

#ifndef VIEWER_H_
#define VIEWER_H_

#include "../app/Types.h"
#include "../data/DataCollector.h"

#include "geometry/BoundingBox.hpp"

#include "display/Renderable.hpp"
#include "display/PointCloud.hpp"

#include <QGLViewer/qglviewer.h>

#include <list>
#include <iostream>
using std::list;

class DataCollector;

enum ProjectionMode { PERSPECTIVE, ORTHOXY, ORTHOXZ, ORTHOYZ};

using lssr::Renderable;
using lssr::BoundingBox;
using lssr::Vertex;

class Viewer : public QGLViewer
{
	Q_OBJECT

public:
	Viewer(QWidget* parent, const QGLWidget* shared = 0);
	virtual ~Viewer();
	virtual void addDataObject(DataCollector* obj);
	virtual void updateDataObject(DataCollector* obj);

	virtual ViewerType type() = 0;
	virtual void centerViewOnObject(Renderable* renderable);

public Q_SLOTS:
	virtual void resetCamera();

protected:
	virtual void draw();

	list<DataCollector*>	    m_dataObjects;
	BoundingBox<Vertex<float> > m_boundingBox;

private:
	QWidget*				    m_parent;
};

#endif /* VIEWER_H_ */
