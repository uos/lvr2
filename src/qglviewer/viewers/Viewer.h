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
 * Viewer.h
 *
 *  Created on: 15.09.2010
 *      Author: Thomas Wiemann
 */

#ifndef VIEWER_H_
#define VIEWER_H_

#include "../app/Types.h"
#include "../data/DataCollector.h"
#include "../widgets/CustomTreeWidgetItem.h"

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

using qglviewer::KeyFrameInterpolator;

class Viewer : public QGLViewer
{
	Q_OBJECT

public:
	Viewer(QWidget* parent, const QGLWidget* shared = 0);
	virtual ~Viewer();
	virtual void addDataObject(DataCollector* obj);
	virtual void removeDataObject(DataCollector* obj);
	void removeDataObject(CustomTreeWidgetItem* item);
	virtual void updateDataObject(DataCollector* obj);

	virtual ViewerType type() = 0;
	virtual void centerViewOnObject(Renderable* renderable);


	KeyFrameInterpolator* kfi() { return m_kfi;}

public Q_SLOTS:
	virtual void resetCamera();
	void createSnapshot();


protected:
	virtual void draw();

	list<DataCollector*>	    m_dataObjects;
	BoundingBox<Vertex<float> > m_boundingBox;
	KeyFrameInterpolator*       m_kfi;


private:
	QWidget*				    m_parent;
};

#endif /* VIEWER_H_ */
