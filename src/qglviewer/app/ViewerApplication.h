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
#include "MeshingOptionsDialogUI.h"

#include "../data/DataCollectorFactory.h"

#include "../viewers/Viewer.h"
#include "../viewers/PerspectiveViewer.h"
#include "../viewers/ViewerManager.h"

#include "../widgets/CustomTreeWidgetItem.h"
#include "../widgets/PointCloudTreeWidgetItem.h"
#include "../widgets/TriangleMeshTreeWidgetItem.h"
#include "../widgets/TransformationDialog.h"
#include "../widgets/DebugOutputDialog.hpp"

#include "display/StaticMesh.hpp"
#include "geometry/HalfEdgeMesh.hpp"

#include "reconstruction/PointCloudManager.hpp"
#include "reconstruction/PCLPointCloudManager.hpp"
#include "reconstruction/StannPointCloudManager.hpp"
#include "reconstruction/FastReconstruction.hpp"



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
	void createMeshFromPointcloud();

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
