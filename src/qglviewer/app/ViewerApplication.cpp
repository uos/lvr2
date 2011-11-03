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
 * ViewerApplication.cpp
 *
 *  Created on: 15.09.2010
 *      Author: Thomas Wiemann
 */

#include "ViewerApplication.h"
#include "../data/Static3DDataCollector.h"

ViewerApplication::ViewerApplication( int argc, char ** argv )
{
	// Setup main window
	m_qMainWindow = new QMainWindow;
	m_mainWindowUi = new MainWindow;
	m_mainWindowUi->setupUi(m_qMainWindow);

	// Add dock widget for currently active objects in viewer
	m_sceneDockWidget = new QDockWidget(m_qMainWindow);
	m_sceneDockWidgetUi = new SceneDockWidgetUI;
	m_sceneDockWidgetUi->setupUi(m_sceneDockWidget);
	m_qMainWindow->addDockWidget(Qt::LeftDockWidgetArea, m_sceneDockWidget);

	// Add tool box widget to dock area
	m_actionDockWidget = new QDockWidget(m_qMainWindow);
	m_actionDockWidgetUi = new ActionDockWidgetUI;
	m_actionDockWidgetUi->setupUi(m_actionDockWidget);
	m_qMainWindow->addDockWidget(Qt::LeftDockWidgetArea, m_actionDockWidget);

	// Setup event manager objects
	m_viewerManager = new ViewerManager(m_qMainWindow);
	m_viewer = m_viewerManager->current();


	m_factory = new DataCollectorFactory;

	// Show window
	m_qMainWindow->show();

	connectEvents();

	/* Load files given as command line arguments. */
	int i;
	for ( i = 1; i < argc; i++ ) {
		printf( "Loading »%s«…\n", argv[i] );
		openFile(string(argv[i]));
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
    QTreeWidget* treeWidget = m_sceneDockWidgetUi->treeWidget;

	// File operations
	QObject::connect(m_mainWindowUi->action_Open , SIGNAL(activated()), this, SLOT(openFile()));

	// Scene views
    connect(m_mainWindowUi->actionShow_entire_scene, SIGNAL(activated()), m_viewer, SLOT(resetCamera()));
    connect(m_mainWindowUi->actionShowSelection, SIGNAL(activated()), this, SLOT(centerOnSelection()));

    // Projections
    connect(m_mainWindowUi->actionXZ_ortho_projection,    SIGNAL(activated()), this, SLOT(setViewerModeOrthoXZ()));
    connect(m_mainWindowUi->actionXY_ortho_projection,    SIGNAL(activated()), this, SLOT(setViewerModeOrthoXY()));
	connect(m_mainWindowUi->actionYZ_ortho_projection,    SIGNAL(activated()), this, SLOT(setViewerModeOrthoYZ()));
    connect(m_mainWindowUi->actionPerspective_projection, SIGNAL(activated()), this, SLOT(setViewerModePerspective()));

	// Render Modes
	connect(m_mainWindowUi->actionVertexView,     SIGNAL(activated()), this, SLOT(meshRenderModeChanged()));
	connect(m_mainWindowUi->actionSurfaceView,    SIGNAL(activated()), this, SLOT(meshRenderModeChanged()));
	connect(m_mainWindowUi->actionWireframeView,  SIGNAL(activated()), this, SLOT(meshRenderModeChanged()));
	connect(m_mainWindowUi->actionPointCloudView, SIGNAL(activated()), this, SLOT(pointRenderModeChanged()));

	// Fog settings
	connect(m_mainWindowUi->actionToggle_fog, SIGNAL(activated()),  this, SLOT(toggleFog()));
	connect(m_mainWindowUi->actionFog_settings, SIGNAL(activated()),this, SLOT(displayFogSettingsDialog()));

	// Communication between the manager objects
    connect(m_factory, SIGNAL(dataCollectorCreated(DataCollector*)), m_viewerManager, SLOT(addDataCollector(DataCollector*)));
    connect(m_factory, SIGNAL(dataCollectorCreated(DataCollector*)), this,            SLOT(dataCollectorAdded(DataCollector*)));

	// Communication between tree widget items
	connect(treeWidget, SIGNAL(itemClicked(QTreeWidgetItem*, int)),         this, SLOT(treeItemClicked(QTreeWidgetItem*, int)));
	connect(treeWidget, SIGNAL(itemChanged(QTreeWidgetItem*, int)),         this, SLOT(treeItemChanged(QTreeWidgetItem*, int)));
	connect(treeWidget, SIGNAL(itemSelectionChanged()),                     this, SLOT(treeSelectionChanged()));
	connect(treeWidget, SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(treeContextMenuRequested(const QPoint &)));

	// Actions
	connect(m_sceneDockWidgetUi->actionExport, SIGNAL(triggered()), this, SLOT(saveSelectedObject()));
	connect(m_mainWindowUi->actionGenerateMesh,               SIGNAL(triggered()), this, SLOT(createMeshFromPointcloud()));

	// Action dock functions
	connect(m_actionDockWidgetUi->buttonCreateMesh, SIGNAL(clicked()), this, SLOT(createMeshFromPointcloud()));
    connect(m_actionDockWidgetUi->buttonTransform,  SIGNAL(clicked()), this, SLOT(transformObject()));
    connect(m_actionDockWidgetUi->buttonDelete,     SIGNAL(clicked()), this, SLOT(deleteObject()));
    connect(m_actionDockWidgetUi->buttonExport,     SIGNAL(clicked()), this, SLOT(saveSelectedObject()));
}

void ViewerApplication::createMeshFromPointcloud()
{
    // Some usings
    using namespace lssr;

    // Display mesh generation dialog
    QTreeWidgetItem* item = m_sceneDockWidgetUi->treeWidget->currentItem();
    if(item)
    {
        if(item->type() == PointCloudItem)
        {
            CustomTreeWidgetItem* c_item = static_cast<CustomTreeWidgetItem*>(item);

            // Create a dialog to parse options
            QDialog* mesh_dialog = new QDialog(m_qMainWindow);
            Ui::MeshingOptionsDialogUI* mesh_ui = new Ui::MeshingOptionsDialogUI;
            mesh_ui->setupUi(mesh_dialog);
            int result = mesh_dialog->exec();

            // Check dialog result and create mesh
            if(result == QDialog::Accepted)
            {
                // Get point cloud data
                PointCloud* pc = static_cast<lssr::PointCloud*>(c_item->renderable());
                PointBuffer* loader = pc->model()->m_pointCloud;

                if(loader)
                {
                    // Create a point cloud manager object
                    PointCloudManager<ColorVertex<float, unsigned char>, Normal<float> >* pcm;
                    QString pcm_name = mesh_ui->comboBoxPCM->currentText();

                    if(pcm_name == "PCL")
                    {
#ifdef _USE_PCL_
                        pcm = new PCLPointCloudManager<ColorVertex<float, unsigned char>, Normal<float> > (loader);
#else
                        pcm = new StannPointCloudManager<ColorVertex<float, unsigned char>, Normal<float> > (loader);
#endif
                    }
                    else
                    {
                        pcm = new StannPointCloudManager<ColorVertex<float, unsigned char>, Normal<float> > (loader);
                    }

                    // Set pcm parameters
                    pcm->setKD(mesh_ui->spinBoxKd->value());
                    pcm->setKI(mesh_ui->spinBoxKi->value());
                    pcm->setKN(mesh_ui->spinBoxKn->value());
                    pcm->calcNormals();

                    // Create an empty mesh
                    HalfEdgeMesh<ColorVertex<float, unsigned char>, Normal<float> > mesh(pcm);

                    // Get reconstruction mesh
                    float voxelsize = mesh_ui->spinBoxVoxelsize->value();

                    FastReconstruction<ColorVertex<float, unsigned char>, Normal<float> > reconstruction(*pcm, voxelsize, true);
                    reconstruction.getMesh(mesh);

                    // Get optimization parameters
                    bool optimize_planes = mesh_ui->checkBoxOptimizePlanes->isChecked();
//                  bool fill_holes      = mesh_ui->checkBoxFillHoles->isChecked();
//                  bool rds             = mesh_ui->checkBoxRDA->isChecked();
//                  bool small_regions   = mesh_ui->checkBoxRemoveRegions->isChecked();
                    bool retesselate     = mesh_ui->checkBoxRetesselate->isChecked();
                    bool texture         = mesh_ui->checkBoxGenerateTextures->isChecked();
                    bool color_regions   = mesh_ui->checkBoxColorRegions->isChecked();

                    int  num_plane_its   = mesh_ui->spinBoxPlaneIterations->value();
//                  int  num_rda         = mesh_ui->spinBoxRDA->value();
                    int  num_rm_regions  = mesh_ui->spinBoxRemoveRegions->value();

                    float min_plane_size = mesh_ui->spinBoxMinPlaneSize->value();
                    float normal_thresh  = mesh_ui->spinBoxNormalThr->value();
                    float max_hole_size  = mesh_ui->spinBoxHoleSize->value();

                    // Perform optimizations
                    if(optimize_planes)
                    {
                        if(color_regions)
                        {
                            mesh.enableRegionColoring();
                        }

                        mesh.optimizePlanes(num_plane_its,
                                normal_thresh,
                                min_plane_size,
                                num_rm_regions,
                                true);

                        mesh.fillHoles(max_hole_size);
                        mesh.optimizePlaneIntersections();
                        mesh.restorePlanes(min_plane_size);
                    }

                    if(retesselate)
                    {
                        mesh.finalizeAndRetesselate(texture);
                    }
                    else
                    {
                        mesh.finalize();
                    }


                    // Create and add mesh to loaded objects
                    MeshBuffer* l = mesh.meshBuffer();

                    lssr::StaticMesh* static_mesh = new lssr::StaticMesh(l);
                    TriangleMeshTreeWidgetItem* mesh_item = new TriangleMeshTreeWidgetItem(TriangleMeshItem);

                    int modes = 0;
                    modes |= Mesh;

                    string name = "Mesh: " + c_item->name();

                    cout << static_mesh->getNumberOfFaces() << endl;
                    cout << static_mesh->getNumberOfVertices() << endl;

                    if(static_mesh->getNormals())
                    {
                        modes |= VertexNormals;
                    }
                    mesh_item->setSupportedRenderModes(modes);
                    mesh_item->setViewCentering(false);
                    mesh_item->setName(name);
                    mesh_item->setRenderable(static_mesh);
                    mesh_item->setNumFaces(static_mesh->getNumberOfFaces());

                    Static3DDataCollector* dc = new Static3DDataCollector(static_mesh, name, mesh_item);

                    dataCollectorAdded(dc);
                    m_viewerManager->addDataCollector(dc);
                }

            }
        }
    }

}

void ViewerApplication::openFile()
{
    QFileDialog file_dialog;
    QStringList file_names;
    QStringList file_types;

    file_types << "Point Clouds (*.pts)"
            //             << "Points and Normals (*.nor)"
            << "PLY Models (*.ply)"
            //             << "Polygonal Meshes (*.bor)"
            << "All Files (*.*)";


    //Set Title
    file_dialog.setWindowTitle("Open File");
    file_dialog.setFileMode(QFileDialog::ExistingFile);
    file_dialog.setFilters(file_types);

    if(file_dialog.exec()){
        file_names = file_dialog.selectedFiles();
    } else {
        return;
    }

    //Get filename from list
    string file_name = file_names.constBegin()->toStdString();
    m_factory->create(file_name);



}

void ViewerApplication::meshRenderModeChanged()
{
    QTreeWidgetItem* item = m_sceneDockWidgetUi->treeWidget->currentItem();
    if(item)
    {
        if(item->type() == TriangleMeshItem)
        {
            TriangleMeshTreeWidgetItem* t_item = static_cast<TriangleMeshTreeWidgetItem*>(item);

            // Setup new render mode
            lssr::StaticMesh* mesh = static_cast<lssr::StaticMesh*>(t_item->renderable());

            int renderMode = 0;

            // Check states of buttons
            if(m_mainWindowUi->actionSurfaceView->isChecked()) renderMode |= lssr::RenderSurfaces;
            if(m_mainWindowUi->actionWireframeView->isChecked()) renderMode |= lssr::RenderTriangles;

            // Set proper render mode and forbid nothing selected
            if(renderMode != 0)
            {
                mesh->setRenderMode(renderMode);
            }

            // Avoid inconsistencies in button toggle states
            m_mainWindowUi->actionSurfaceView->setChecked(mesh->getRenderMode() & lssr::RenderSurfaces);
            m_mainWindowUi->actionWireframeView->setChecked(mesh->getRenderMode() & lssr::RenderTriangles);

            // Force redisplay
            m_viewer->updateGL();
        }
    }
}

void ViewerApplication::pointRenderModeChanged()
{

}

void ViewerApplication::openFile(string filename)
{
    m_factory->create(filename);
}

void ViewerApplication::transformObject()
{
    QTreeWidgetItem* item = m_sceneDockWidgetUi->treeWidget->currentItem();
    if(item)
    {
        if(item->type() > 1000)
        {
            CustomTreeWidgetItem* c_item = static_cast<CustomTreeWidgetItem*>(item);
//            TransformationDialog* d = 
			new TransformationDialog(m_viewer, c_item->renderable());
        }
    }
}

void ViewerApplication::deleteObject()
{
    // Ask in Microsoft stype ;-)
    QMessageBox msgBox;
    msgBox.setText("Remove selected object?");
    msgBox.setStandardButtons(QMessageBox::Ok | QMessageBox::Cancel );
    msgBox.setDefaultButton(QMessageBox::Cancel);
    int ret = msgBox.exec();

    // Only delete objects when uses says "Yeah man, ok! Do It!!"
    if(ret == QMessageBox::Ok)
    {
        QTreeWidgetItem* item = m_sceneDockWidgetUi->treeWidget->currentItem();
        if(item)
        {
            if(item->type() > 1000)
            {
                // Remove item from tree widget
                CustomTreeWidgetItem* c_item = static_cast<CustomTreeWidgetItem*>(item);
                int i = m_sceneDockWidgetUi->treeWidget->indexOfTopLevelItem(item);
                m_sceneDockWidgetUi->treeWidget->takeTopLevelItem(i);

                // Delete  data collector
                m_viewerManager->current()->removeDataObject(c_item);

            }
        }
    }
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
            QAction* mesh_action = m_mainWindowUi->actionGenerateMesh;
            actions.append(mesh_action);
        }

        if(item->type() == PointCloudItem)
        {
            QAction* mesh_action = m_mainWindowUi->actionGenerateMesh;
            actions.append(mesh_action);
        }

        // Add standard action to context menu
        actions.append(m_mainWindowUi->actionShowSelection);
        actions.append(m_sceneDockWidgetUi->actionExport);
    }

    // Display menu if actions are present
    if (actions.count() > 0)
    {
       QMenu::exec(actions, m_sceneDockWidgetUi->treeWidget->mapToGlobal(position));
    }

}

void ViewerApplication::saveSelectedObject()
{
    QTreeWidgetItem* item = m_sceneDockWidgetUi->treeWidget->currentItem();
    if(item)
    {

        QFileDialog file_dialog;
        QStringList file_names;
        QStringList file_types;

        // Parse extensions by file type
        if(item->type() == PointCloudItem || item->type() == MultiPointCloudItem)
        {
            file_types << "Point Clouds (*.pts)"
                       << "PLY Models (*.ply)"
                       << "All Files (*.*)";
        }
        else if (item->type() == TriangleMeshItem)
        {
            file_types << "PLY Models (*.ply)"
                       << "OBJ Models (*.obj)"
                       << "All Files (*.*)";
        }
        else
        {
            QMessageBox msgBox;
            msgBox.setText("Object type not supported");
            msgBox.setStandardButtons(QMessageBox::Ok);
            return;
        }

        //Set Title
        file_dialog.setWindowTitle("Save selected object");
        file_dialog.setFilters(file_types);

        if(file_dialog.exec()){
            file_names = file_dialog.selectedFiles();
        } else {
            return;
        }

        string file_name = file_names.constBegin()->toStdString();

        // Cast to custom item
        if(item->type() > 1000)
        {
            CustomTreeWidgetItem* c_item = static_cast<CustomTreeWidgetItem*>(item);

            lssr::Model* m = c_item->renderable()->model();
            lssr::ModelFactory::saveModel(m, file_name);

        }

    }
}

void ViewerApplication::dataCollectorAdded(DataCollector* d)
{
    if(d->treeItem())
    {
        m_sceneDockWidgetUi->treeWidget->addTopLevelItem(d->treeItem());
        updateToolbarActions(d->treeItem());
        updateActionDock(d->treeItem());
    }
}

void ViewerApplication::treeItemClicked(QTreeWidgetItem* item, int d)
{
    cout << "Clicked" << endl;

    // Center view on selected item if enabled
    if(item->type() > 1000)
    {
        CustomTreeWidgetItem* custom_item = static_cast<CustomTreeWidgetItem*>(item);
        if(custom_item->centerOnClick())
        {
            m_viewer->centerViewOnObject(custom_item->renderable());
            updateToolbarActions(custom_item);
            updateActionDock(custom_item);
        }
    }

    // Parse special operations of different items
}

void ViewerApplication::treeItemChanged(QTreeWidgetItem* item, int d)
{
    cout << "changed" << endl;

    if(item->type() > 1000)
    {
        CustomTreeWidgetItem* custom_item = static_cast<CustomTreeWidgetItem*>(item);
        custom_item->renderable()->setActive(custom_item->checkState(d) == Qt::Checked);
        m_viewer->updateGL();
    }
}


void ViewerApplication::treeSelectionChanged()
{

    // Unselect all custom items
    QTreeWidgetItemIterator w_it( m_sceneDockWidgetUi->treeWidget);
    while (*w_it)
    {
        if( (*w_it)->type() >= ServerItem)
        {
            CustomTreeWidgetItem* item = static_cast<CustomTreeWidgetItem*>(*w_it);
            item->renderable()->setSelected(false);
        }
        ++w_it;
    }

    QList<QTreeWidgetItem *> list = m_sceneDockWidgetUi->treeWidget->selectedItems();
    QList<QTreeWidgetItem *>::iterator it = list.begin();

    for(it = list.begin(); it != list.end(); it++)
    {
        if( (*it)->type() >= ServerItem)
        {
            // Get selected item
            CustomTreeWidgetItem* item = static_cast<CustomTreeWidgetItem*>(*it);
            item->renderable()->setSelected(true);

            // Update render modes in tool bar
            updateToolbarActions(item);
            updateActionDock(item);

        }
    }

    m_viewer->updateGL();
}

void ViewerApplication::updateToolbarActions(CustomTreeWidgetItem* item)
{
//    bool point_support = item->supportsMode(Points);
//    bool pn_support = item->supportsMode(PointNormals);
//    bool vn_support = item->supportsMode(VertexNormals);
    bool mesh_support = item->supportsMode(Mesh);

    if(mesh_support)
    {
        m_mainWindowUi->actionVertexView->setEnabled(true);
        m_mainWindowUi->actionWireframeView->setEnabled(true);
        m_mainWindowUi->actionSurfaceView->setEnabled(true);
        m_mainWindowUi->actionPointCloudView->setEnabled(false);
        m_mainWindowUi->actionGenerateMesh->setEnabled(false);
    }
    else
    {
        m_mainWindowUi->actionVertexView->setEnabled(false);
        m_mainWindowUi->actionWireframeView->setEnabled(false);
        m_mainWindowUi->actionSurfaceView->setEnabled(false);
        m_mainWindowUi->actionPointCloudView->setEnabled(true);
        m_mainWindowUi->actionGenerateMesh->setEnabled(true);
    }

}

void ViewerApplication::updateActionDock(CustomTreeWidgetItem* item)
{
       m_actionDockWidgetUi->buttonCreateMesh->setEnabled(item->supportsMode(Points));
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
		(static_cast<PerspectiveViewer*>(m_viewer))->setFogType(FOG_LINEAR);
	}
}

void ViewerApplication::fogExp2()
{
	if(m_viewer->type() == PERSPECTIVE_VIEWER)
	{
		(static_cast<PerspectiveViewer*>(m_viewer))->setFogType(FOG_EXP2);
	}
}

void ViewerApplication::fogExp()
{
	if(m_viewer->type() == PERSPECTIVE_VIEWER)
	{
		(static_cast<PerspectiveViewer*>(m_viewer))->setFogType(FOG_EXP);
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

void ViewerApplication::centerOnSelection()
{
    QTreeWidgetItem* item = m_sceneDockWidgetUi->treeWidget->currentItem();
    if(item)
    {
        if(item->type() > 1000)
        {
            CustomTreeWidgetItem* c_item = static_cast<CustomTreeWidgetItem*>(item);
            m_viewer->centerViewOnObject(c_item->renderable());
        }
    }
}

ViewerApplication::~ViewerApplication()
{
	//if(m_qMainWindow != 0) delete m_qMainWindow;
	//if(m_mainWindowUI != 0) delete m_mainWindowUI;
	if(m_viewer != 0) delete m_viewer;
}
