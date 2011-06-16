/*
 * ViewerApplication.h
 *
 *  Created on: 15.09.2010
 *      Author: Thomas Wiemannn
 */

#ifndef VIEWERAPPLICATION_H_
#define VIEWERAPPLICATION_H_

#include <QtGui>

#include "MainWindow.h"
#include "FogDensityDialog.h"
#include "SceneDockWidget.h"

#include "../data/DataManager.h"

#include "../viewers/Viewer.h"
#include "../viewers/PerspectiveViewer.h"
#include "../viewers/ViewerManager.h"

#include "../widgets/CustomTreeWidgetItem.h"


using Ui::MainWindow;
using Ui::Fogsettings;
using Ui::SceneDockWidget;

class EventManager;

class ViewerApplication : public QObject
{
	Q_OBJECT

public:
	ViewerApplication(int argc, char** argv);
	virtual ~ViewerApplication();

public Q_SLOTS:
	void setViewerModePerspective();
	void setViewerModeOrthoXY();
	void setViewerModeOrthoXZ();
	void setViewerModeOrthoYZ();
	void toggleFog();
	void displayFogSettingsDialog();
	void fogDensityChanged(int i);
	void fogLinear();
	void fogExp2();
	void fogExp();

	void dataCollectorAdded(DataCollector* d);

private:

	void connectEvents();

	MainWindow*					m_mainWindowUi;
	QMainWindow*				m_qMainWindow;

	Viewer*						m_viewer;
	QDialog*					m_fogSettingsDialog;
	SceneDockWidget*			m_sceneDockWidgetUi;
	QDockWidget*				m_sceneDockWidget;

	Fogsettings*				m_fogSettingsUI;
	ViewerManager*				m_viewerManager;
	DataManager*				m_dataManager;
};

#endif /* VIEWERAPPLICATION_H_ */
