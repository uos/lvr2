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

/**
 * MainWindow.hpp
 *
 *  @date Jan 31, 2014
 *  @author Thomas Wiemann
 */
#ifndef MAINWINDOW_HPP_
#define MAINWINDOW_HPP_

#include <vtkSmartPointer.h>
#include <vtkCamera.h>
#include <vtkCameraRepresentation.h>
#include <vtkCameraInterpolator.h>
#include <vtkCommand.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkViewport.h>
#include <vtkObjectFactory.h>
#include <vtkGraphicsFactory.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkAxesActor.h>

#include <QtGui>
#include "LVRMainWindowUI.h"
#include "LVRAboutDialogUI.h"
#include "LVRTreeWidgetHelper.hpp"
#include "../vtkBridge/LVRModelBridge.hpp"
#include "../widgets/LVRModelItem.hpp"
#include "../widgets/LVRPointCloudItem.hpp"
#include "../widgets/LVRMeshItem.hpp"
#include "../widgets/LVRItemTypes.hpp"
#include "../widgets/LVRRenameDialog.hpp"
#include "../widgets/LVRAnimationDialog.hpp"
#include "../widgets/LVRTransformationDialog.hpp"
#include "../widgets/LVRCorrespondanceDialog.hpp"
#include "../widgets/LVRReconstructionEstimateNormalsDialog.hpp"
#include "../widgets/LVRReconstructionMarchingCubesDialog.hpp"
#include "../widgets/LVRReconstructionExtendedMarchingCubesDialog.hpp"
#include "../widgets/LVROptimizationPlanarOptimizationDialog.hpp"
#include "../widgets/LVROptimizationRemoveArtifactsDialog.hpp"
#include "../widgets/LVRFilteringMLSProjectionDialog.hpp"
#include "../widgets/LVRFilteringRemoveOutliersDialog.hpp"
#include "../widgets/LVRBackgroundDialog.hpp"

#include "../vtkBridge/LVRPickingInteractor.hpp"
#include "../vtkBridge/LVRVtkArrow.hpp"

#include <iostream>
#include <iterator>
#include <vector>
#include "boost/format.hpp"
using std::vector;
using std::cout;
using std::endl;

namespace lvr
{

class LVRMainWindow : public QMainWindow, public Ui::MainWindow
{
    Q_OBJECT
public:

    /**
     * @brief   MainWindow
     */
    LVRMainWindow();
    virtual ~LVRMainWindow();

public Q_SLOTS:
    void loadModel();
    void manualICP();
    void showTransformationDialog();
    void showTreeContextMenu(const QPoint&);
    void showColorDialog();
    void showAboutDialog(QAction*);
    void renameModelItem();
    void estimateNormals();
    void reconstructUsingMarchingCubes();
    void reconstructUsingExtendedMarchingCubes();
    void reconstructUsingPlanarMarchingCubes();
    void optimizePlanes();
    void removeArtifacts();
    void applyMLSProjection();
    void removeOutliers();
    void deleteModelItem();
    void changePointSize(int pointSize);
    void changeTransparency(int transparencyValue);
    void changeShading(int shader);
    void assertToggles();
    void togglePoints(bool checkboxState);
    void toggleNormals(bool checkboxState);
    void toggleMeshes(bool checkboxState);
    void toggleWireframe(bool checkboxState);
    void refreshView();
    void updateView();
    void saveCamera();
    void loadCamera();
    void parseCommandLine(int argc, char** argv);
    void openCameraPathTool();
    void removeArrow(LVRVtkArrow*);
    void addArrow(LVRVtkArrow*);
    void alignPointClouds();
    void exportSelectedModel();
    void buildIncompatibilityBox(string actionName, unsigned char allowedTypes);
    void showBackgroundDialog();

    LVRModelItem* getModelItem(QTreeWidgetItem* item);
    LVRPointCloudItem* getPointCloudItem(QTreeWidgetItem* item);
    LVRMeshItem* getMeshItem(QTreeWidgetItem* item);

protected Q_SLOTS:
    void setModelVisibility(QTreeWidgetItem* treeWidgetItem, int column);
    void restoreSliders(QTreeWidgetItem* treeWidgetItem, int column);

Q_SIGNALS:
    void correspondenceDialogOpened();

private:
    void setupQVTK();
    void connectSignalsAndSlots();

    LVRCorrespondanceDialog*                    m_correspondanceDialog;
    QDialog*                                    m_aboutDialog;
    QMessageBox*                                m_incompatibilityBox;
    vtkSmartPointer<vtkRenderer>                m_renderer;
    vtkSmartPointer<vtkRenderWindowInteractor>  m_renderWindowInteractor;
    vtkSmartPointer<vtkCamera>			        m_camera;
    vtkSmartPointer<vtkCameraRepresentation>    m_pathCamera;
    vtkSmartPointer<vtkOrientationMarkerWidget> m_axesWidget;
    vtkSmartPointer<vtkAxesActor> 				m_axes;

    QMenu*				                        m_treeParentItemContextMenu;
    QMenu*                                      m_treeChildItemContextMenu;

    // Toolbar item "File"
	QAction*							m_actionOpen;
	QAction*							m_actionExport;
	QAction*							m_actionQuit;
	// Toolbar item "Views"
	QAction*							m_actionReset_Camera;
	QAction*							m_actionStore_Current_View;
	QAction*							m_actionRecall_Stored_View;
	QAction*                            m_actionCameraPathTool;
    // Toolbar item "Reconstruction"
	QAction*                            m_actionEstimate_Normals;
	QAction*                            m_actionMarching_Cubes;
    QAction*                            m_actionPlanar_Marching_Cubes;
    QAction*                            m_actionExtended_Marching_Cubes;
    QAction*                            m_actionCompute_Textures;
    QAction*                            m_actionMatch_Textures_from_Package;
    QAction*                            m_actionExtract_and_Rematch_Patterns;
    // Toolbar item "Mesh Optimization"
    QAction*                            m_actionPlanar_Optimization;
    QAction*                            m_actionRemove_Artifacts;
    // Toolbar item "Filtering"
    QAction*                            m_actionRemove_Outliers;
    QAction*                            m_actionMLS_Projection;
    // Toolbar item "Registration"
    QAction*                            m_actionICP_Using_Manual_Correspondance;
    QAction*                            m_actionICP_Using_Pose_Estimations;
    QAction*                            m_actionGlobal_Relaxation;
    // Toolbar item "Classification"
    QAction*                            m_actionSimple_Plane_Classification;
    QAction*                            m_actionFurniture_Recognition;
	// Toolbar item "About"
	QMenu*                              m_menuAbout;
	// QToolbar below toolbar
	QAction*							m_actionShow_Points;
	QAction*							m_actionShow_Normals;
	QAction*							m_actionShow_Mesh;
	QAction*							m_actionShow_Wireframe;
	QAction*                            m_actionShowBackgroundSettings;
    // Sliders below tree widget
    QSlider*							m_horizontalSliderPointSize;
    QSlider*							m_horizontalSliderTransparency;
    // Combo boxes below sliders
    QComboBox*							m_comboBoxGradient;
    QComboBox*							m_comboBoxShading;
    // Buttons below combo boxes
    QPushButton*						m_buttonCameraPathTool;
    QPushButton*						m_buttonCreateMesh;
    QPushButton*						m_buttonExportData;
    QPushButton*						m_buttonTransformModel;

	QAction*				            m_actionShowColorDialog;
    QAction*                            m_actionRenameModelItem;
	QAction*			                m_actionDeleteModelItem;
	QAction*                            m_actionExportModelTransformed;

    LVRPickingInteractor*               m_pickingInteractor;
    LVRTreeWidgetHelper*                m_treeWidgetHelper;

    enum TYPE {
        MODELITEMS_ONLY,
        POINTCLOUDS_ONLY,
        MESHES_ONLY,
        POINTCLOUDS_AND_PARENT_ONLY,
        MESHES_AND_PARENT_ONLY,
        POINTCLOUDS_AND_MESHES_ONLY,
        POINTCLOUDS_AND_MESHES_AND_PARENT_ONLY
    };
};

} /* namespace lvr */

#endif /* MAINWINDOW_HPP_ */
