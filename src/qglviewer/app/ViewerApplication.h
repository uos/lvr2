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

#include "../data/DataCollectorFactory.h"

#include "../viewers/Viewer.h"
#include "../viewers/PerspectiveViewer.h"
#include "../viewers/ViewerManager.h"

#include "../widgets/CustomTreeWidgetItem.h"
#include "../widgets/PointCloudTreeWidgetItem.h"
#include "../widgets/TriangleMeshTreeWidgetItem.h"
#include "../widgets/TransformationDialog.h"

#include "display/StaticMesh.hpp"


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
	void treeItemClicked(QTreeWidgetItem* item, int n);
	void treeItemChanged(QTreeWidgetItem*, int);
	void treeSelectionChanged();
	void treeContextMenuRequested(const QPoint &);
	void treeWidgetExport();
	void transformObject();

	void openFile();

	void meshRenderModeChanged();
	void pointRenderModeChanged();

private:

	void updateToolbarActions(CustomTreeWidgetItem* item);
	void connectEvents();
	void openFile(string filename);

	MainWindow*					m_mainWindowUi;
	QMainWindow*				m_qMainWindow;

	Viewer*						m_viewer;
	QDialog*					m_fogSettingsDialog;
	SceneDockWidget*			m_sceneDockWidgetUi;
	QDockWidget*				m_sceneDockWidget;

	Fogsettings*				m_fogSettingsUI;
	ViewerManager*				m_viewerManager;
	DataCollectorFactory*       m_factory;
};

#endif /* VIEWERAPPLICATION_H_ */
