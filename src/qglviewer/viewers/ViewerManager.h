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
