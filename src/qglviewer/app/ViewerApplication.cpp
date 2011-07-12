/*
 * ViewerApplication.cpp
 *
 *  Created on: 15.09.2010
 *      Author: Thomas Wiemann
 */

#include "ViewerApplication.h"

ViewerApplication::ViewerApplication( int argc, char ** argv )
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

	/* Load files given as command line arguments. */
	int i;
	for ( i = 1; i < argc; i++ ) {
		printf( "Loading »%s«…\n", argv[i] );
		m_dataManager->loadFile( argv[i] );
	}

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

    QObject::connect(m_dataManager, SIGNAL(dataCollectorCreated(DataCollector*)),
                    this, SLOT(dataCollectorAdded(DataCollector*)));

	QObject::connect(m_dataManager, SIGNAL(dataCollectorUpdate(DataCollector*)),
					m_viewerManager, SLOT(updateDataObject(DataCollector*)));

	QObject::connect(m_sceneDockWidgetUi->treeWidget, SIGNAL(itemClicked(QTreeWidgetItem*, int)),
					this, SLOT(treeItemClicked(QTreeWidgetItem*, int)));

	QObject::connect(m_sceneDockWidgetUi->treeWidget, SIGNAL(itemChanged(QTreeWidgetItem*, int)),
	                    this, SLOT(treeItemChanged(QTreeWidgetItem*, int)));

	QObject::connect(m_sceneDockWidgetUi->treeWidget, SIGNAL(itemSelectionChanged()),
	                    this, SLOT(treeSelectionChanged()));

	QObject::connect(m_sceneDockWidgetUi->treeWidget, SIGNAL(customContextMenuRequested(const QPoint &)),
	                        this, SLOT(treeContextMenuRequested(const QPoint &)));

	// Context menu actions
	connect(m_sceneDockWidgetUi->actionExport_selected_scans, SIGNAL(triggered()), this, SLOT(treeWidgetExport()));

}

void ViewerApplication::treeContextMenuRequested(const QPoint &position)
{
    // Create suitable actions for clicked widget
    QList<QAction *> actions;
    QTreeWidgetItem* item = m_sceneDockWidgetUi->treeWidget->itemAt(position);

    // Check if item is valid and parse supported actions
    if(item)
    {
        if(item->type() == MultiPointCloudItem)
        {
            QAction* action = m_sceneDockWidgetUi->actionExport_selected_scans;

            actions.append(action);
        }
    }


    // Display menu if actions are present
    if (actions.count() > 0)
    {
       QMenu::exec(actions, m_sceneDockWidgetUi->treeWidget->mapToGlobal(position));
    }

}

void ViewerApplication::treeWidgetExport()
{
    cout << "Export" << endl;
}

void ViewerApplication::dataCollectorAdded(DataCollector* d)
{
    if(d->treeItem())
    {
        m_sceneDockWidgetUi->treeWidget->addTopLevelItem(d->treeItem());
    }
}

void ViewerApplication::treeItemClicked(QTreeWidgetItem* item, int d)
{
    // Center view on selected item if enabled
    if(item->type() > 1000)
    {
        CustomTreeWidgetItem* custom_item = static_cast<CustomTreeWidgetItem*>(item);
        if(custom_item->centerOnClick())
        {
            m_viewer->centerViewOnObject(custom_item->renderable());
        }
    }

    // Parse special operations of diffrent items
}

void ViewerApplication::treeItemChanged(QTreeWidgetItem* item, int d)
{
    if(item->type() == PointCloudItem)
    {
        CustomTreeWidgetItem* custom_item = static_cast<CustomTreeWidgetItem*>(item);
        custom_item->renderable()->setActive(custom_item->checkState(d) == Qt::Checked);
        m_viewer->updateGL();
    }
}


void ViewerApplication::treeSelectionChanged()
{
    QTreeWidgetItemIterator it(m_sceneDockWidgetUi->treeWidget);
    while (*it) {
        if( (*it)->type() >= ServerItem)
        {
           CustomTreeWidgetItem* item = static_cast<CustomTreeWidgetItem*>(*it);
           item->renderable()->setSelected(item->isSelected());
        }
        ++it;
    }
    m_viewer->updateGL();
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
