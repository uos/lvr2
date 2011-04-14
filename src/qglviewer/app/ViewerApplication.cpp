/*
 * ViewerApplication.cpp
 *
 *  Created on: 15.09.2010
 *      Author: Thomas Wiemann
 */

#include "ViewerApplication.h"

#include "../widgets/ServerTreeWidgetItem.h"
#include "../widgets/InterfaceTreeWidgetItem.h"

ViewerApplication::ViewerApplication(int argc, char** argv)
{
	// Setup main window
	m_qMainWindow = new QMainWindow;
	m_mainWindowUi = new MainWindow;
	m_mainWindowUi->setupUi(m_qMainWindow);

	// Add dock widget for currently active objects in viewer
	m_sceneDockWidget = new QDockWidget(m_qMainWindow);
	m_sceneDockWidgetUi = new SceneDockWidget;
	m_sceneDockWidgetUi->setupUi(m_sceneDockWidget);
	m_qMainWindow->addDockWidget(Qt::LeftDockWidgetArea, m_sceneDockWidget);

	// Setup event manager objects
	m_viewerManager = new ViewerManager(m_qMainWindow);
	m_viewer = m_viewerManager->current();

	m_dataManager = new DataManager;


	// Show window
	m_qMainWindow->show();

	connectEvents();

	// Call a resize to fit viewers to their parent widgets
	m_viewer->setGeometry(m_qMainWindow->centralWidget()->geometry());
	m_viewer->setBackgroundColor(QColor(255, 255, 255));
	m_qMainWindow->setCentralWidget(m_viewer);

	// Initalize dialogs
	m_fogSettingsUI = 0;
	m_fogSettingsDialog = 0;
}

void ViewerApplication::connectEvents()
{

	// File operations
	QObject::connect(m_mainWindowUi->action_Open , SIGNAL(activated()),
			m_dataManager, SLOT(openFile()));

	// Player actions
	QObject::connect(m_mainWindowUi->actionConnect_to_server, SIGNAL(activated()),
			this, SLOT(addPlayerServer()));

	// Projection settings
	QObject::connect(m_mainWindowUi->actionShow_entire_scene, SIGNAL(activated()),
			m_viewer, SLOT(resetCamera()));
	QObject::connect(m_mainWindowUi->actionXZ_ortho_projection, SIGNAL(activated()),
			this, SLOT(setViewerModeOrthoXZ()));
	QObject::connect(m_mainWindowUi->actionXY_ortho_projection, SIGNAL(activated()),
			this, SLOT(setViewerModeOrthoXY()));
	QObject::connect(m_mainWindowUi->actionYZ_ortho_projection, SIGNAL(activated()),
			this, SLOT(setViewerModeOrthoYZ()));
	QObject::connect(m_mainWindowUi->actionPerspective_projection, SIGNAL(activated()),
			this, SLOT(setViewerModePerspective()));

	// Fog settings
	QObject::connect(m_mainWindowUi->actionToggle_fog, SIGNAL(activated()),
			this, SLOT(toggleFog()));
	QObject::connect(m_mainWindowUi->actionFog_settings, SIGNAL(activated()),
				this, SLOT(displayFogSettingsDialog()));

	// Communication between the manager objects
	QObject::connect(m_dataManager, SIGNAL(dataCollectorCreated(DataCollector*)),
					m_viewerManager, SLOT(addDataCollector(DataCollector*)));

	QObject::connect(m_dataManager, SIGNAL(dataCollectorUpdate(DataCollector*)),
					m_viewerManager, SLOT(updateDataObject(DataCollector*)));

	QObject::connect(m_sceneDockWidgetUi->treeWidget, SIGNAL(itemClicked(QTreeWidgetItem*, int)),
					this, SLOT(treeItemClicked(QTreeWidgetItem*, int)));


}

void ViewerApplication::toggleFog()
{
	if(m_viewer->type() == PERSPECTIVE_VIEWER)
	{
		(static_cast<PerspectiveViewer*>(m_viewer))->toggleFog();
	}
}

void ViewerApplication::displayFogSettingsDialog()
{
	if(!m_fogSettingsUI)
	{
		m_fogSettingsUI = new Fogsettings;
		m_fogSettingsDialog = new QDialog(m_qMainWindow);
		m_fogSettingsUI->setupUi(m_fogSettingsDialog);

//		QObject::connect(m_fogSettingsUI->sliderDensity, SIGNAL(valueChanged(int)),
//						this, SLOT(fogDensityChanged(int)));
		QObject::connect(m_fogSettingsUI->sliderDensity, SIGNAL(sliderMoved(int)),
						this, SLOT(fogDensityChanged(int)));
		QObject::connect(m_fogSettingsUI->radioButtonLinear , SIGNAL(clicked()),
						this, SLOT(fogLinear()));
		QObject::connect(m_fogSettingsUI->radioButtonExp , SIGNAL(clicked()),
						this, SLOT(fogExp()));
		QObject::connect(m_fogSettingsUI->radioButtonExp2 , SIGNAL(clicked()),
						this, SLOT(fogExp2()));


	}

	m_fogSettingsDialog->show();
	m_fogSettingsDialog->raise();
	m_fogSettingsDialog->activateWindow();

}

void ViewerApplication::addPlayerServer()
{
	// Show dialog
	QDialog dialog;
	PlayerConnectionDialog* dialog_ui = new PlayerConnectionDialog;
	dialog_ui->setupUi(&dialog);
	int answer = dialog.exec();

	if(answer == QDialog::Accepted)
	{
		// Parse dialog data
		string ip = dialog_ui->lineEditHost->text().toStdString();
		int port = dialog_ui->spinBoxPort->value();

		// Try to get a server connection
		PlayerConnectionManager* manager = PlayerConnectionManager::instance();

		PlayerServer* server = manager->addServer(ip, port);
		if(server != 0)
		{
			ServerTreeWidgetItem* item = new ServerTreeWidgetItem(server);
			m_sceneDockWidgetUi->treeWidget->addTopLevelItem(item);
			item->setInitialState(Qt::Checked);
		}
		else
		{
			cout << "(1) Could not connect" << endl;
		}
  	}

}

void ViewerApplication::treeItemClicked(QTreeWidgetItem* qitem, int col)
{
	// Don't use col, just get rid of compiler warnings ;-)
	col = 0;
	CustomTreeWidgetItem* item = static_cast<CustomTreeWidgetItem*>(qitem);

	// Don't send any duplicate notifications
	if(item->toggled())
	{
		if(item->type() == ServerItem)
		{
			serverTreeItemToggled(qitem);
		}
		else if(item->type() == InterfaceItem)
		{
			interfaceTreeItemToggled(qitem);
		}
	}
}

void ViewerApplication::serverTreeItemToggled(QTreeWidgetItem* qitem)
{
	// Get check state
	Qt::CheckState state = qitem->checkState(0);
	ServerTreeWidgetItem* server_item = static_cast<ServerTreeWidgetItem*>(qitem);

	if(state == Qt::Unchecked)
	{
		int res = QMessageBox::warning(m_qMainWindow, tr("Viewer"),
				tr("Do you really want to close the selected connection?"),
				QMessageBox::No | QMessageBox::Yes);

		if(res == QMessageBox::Yes)
		{
			// Close connection
			ServerTreeWidgetItem* server_item = static_cast<ServerTreeWidgetItem*>(qitem);
			PlayerConnectionManager::instance()->removeServer(server_item->server());

			// Don't know why there is no removeTopLevelItem method
			// in QTreeWidget...
			QTreeWidget* w = m_sceneDockWidgetUi->treeWidget;
			int index = w->indexOfTopLevelItem(qitem);
			w->takeTopLevelItem(index);
		}
		else
		{
			// Restore check state
			server_item->setInitialState(Qt::Checked);
		}
	}
}

void ViewerApplication::interfaceTreeItemToggled(QTreeWidgetItem* qitem)
{
	// Get check state
	Qt::CheckState state = qitem->checkState(0);

	InterfaceTreeWidgetItem* item = static_cast<InterfaceTreeWidgetItem*>(qitem);

	if(state == Qt::Checked)
	{
		// Get server and interface description
		PlayerServer* server = item->server();
		if(server != 0)
		{
			InterfaceDcr dcr = item->description();

			// Create a new proyxy
			ClientProxy* proxy;
			PlayerConnectionManager::instance()->subscribe(proxy,
					server->getHost(), server->getPort(), dcr.id, dcr.index);
			if(proxy != 0)
			{
				m_dataManager->createDataCollector(proxy);
			}
		}
		else
		{
			item->setInitialState(Qt::Unchecked);
		}
	}
	else
	{
		// Get server and interface description
		PlayerServer* server = item->server();
		if(server != 0)
		{
			InterfaceDcr dcr = item->description();
			PlayerConnectionManager::instance()->unsubscribe(server->getHost(),
					server->getPort(), dcr.id, dcr.index);
		}
		else
		{
			item->setInitialState(Qt::Checked);
		}
	}
}

void ViewerApplication::fogDensityChanged(int i)
{
	if(m_viewer->type() == PERSPECTIVE_VIEWER)
	{
		(static_cast<PerspectiveViewer*>(m_viewer))->setFogDensity(1.0f * i / 2000.0f);
	}
}

void ViewerApplication::fogLinear()
{
	if(m_viewer->type() == PERSPECTIVE_VIEWER)
	{
		(static_cast<PerspectiveViewer*>(m_viewer))->setFogType(LINEAR);
	}
}

void ViewerApplication::fogExp2()
{
	if(m_viewer->type() == PERSPECTIVE_VIEWER)
	{
		(static_cast<PerspectiveViewer*>(m_viewer))->setFogType(EXP2);
	}
}

void ViewerApplication::fogExp()
{
	if(m_viewer->type() == PERSPECTIVE_VIEWER)
	{
		(static_cast<PerspectiveViewer*>(m_viewer))->setFogType(EXP);
	}
}

void ViewerApplication::setViewerModePerspective()
{
	if(m_viewer->type() == PERSPECTIVE_VIEWER)
	{
		(static_cast<PerspectiveViewer*>(m_viewer))->setProjectionMode(PERSPECTIVE);
	}
}

void ViewerApplication::setViewerModeOrthoXY()
{
	if(m_viewer->type() == PERSPECTIVE_VIEWER)
	{
		(static_cast<PerspectiveViewer*>(m_viewer))->setProjectionMode(ORTHOXY);
	}
}

void ViewerApplication::setViewerModeOrthoXZ()
{
	if(m_viewer->type() == PERSPECTIVE_VIEWER)
	{
		(static_cast<PerspectiveViewer*>(m_viewer))->setProjectionMode(ORTHOXZ);
	}
}

void ViewerApplication::setViewerModeOrthoYZ()
{
	if(m_viewer->type() == PERSPECTIVE_VIEWER)
	{
		(static_cast<PerspectiveViewer*>(m_viewer))->setProjectionMode(ORTHOYZ);
	}
}

ViewerApplication::~ViewerApplication()
{
	//if(m_qMainWindow != 0) delete m_qMainWindow;
	//if(m_mainWindowUI != 0) delete m_mainWindowUI;
	if(m_viewer != 0) delete m_viewer;
}
