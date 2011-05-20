/*
 * Viewer.h
 *
 *  Created on: 15.09.2010
 *      Author: Thomas Wiemann
 */

#ifndef VIEWER_H_
#define VIEWER_H_

#include "../app/Types.h"

#include "../../lib/model3d/BoundingBox.h"

#include <QGLViewer/qglviewer.h>
#include <list>
using std::list;

class DataCollector;

enum ProjectionMode { PERSPECTIVE, ORTHOXY, ORTHOXZ, ORTHOYZ};

class Viewer : public QGLViewer
{
	Q_OBJECT
public:
	Viewer(QWidget* parent, const QGLWidget* shared = 0);
	virtual ~Viewer();
	virtual void addDataObject(DataCollector* obj);
	virtual void updateDataObject(DataCollector* obj);

	virtual ViewerType type() = 0;

public Q_SLOTS:
	virtual void resetCamera();

protected:
	virtual void draw();

	list<DataCollector*>	m_dataObjects;
	BoundingBox				m_boundingBox;

private:
	QWidget*				m_parent;
};

#endif /* VIEWER_H_ */
