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
 * ViewerManager.h
 *
 *  Created on: 07.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef VIEWERMANAGER_H_
#define VIEWERMANAGER_H_

#include "Viewer.h"
#include "../data/DataCollector.h"

#include <QtGui>
#include <list>
using std::list;

class ViewerManager : public QObject
{
	Q_OBJECT

public:
	ViewerManager(QWidget* parent, const QGLWidget* shared = 0);
	virtual ~ViewerManager();

	Viewer* current();

public Q_SLOTS:
	void addDataCollector(DataCollector* c);
	void updateDataObject(DataCollector* obj);

private:
	Viewer* 			m_currentViewer;
	list<Viewer*> 		m_allViewers;
	QWidget*			m_parentWidget;

};

#endif /* VIEWERMANAGER_H_ */
