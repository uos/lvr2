/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * MainWindow.cpp
 *
 *  @date Jan 31, 2014
 *  @author Thomas Wiemann
 */

#include <QFileInfo>
#include <QAbstractItemView>
#include <QtGui>

#include "LVRMainWindow.hpp"

#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/DataStruct.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/descriptions/HDF5Kernel.hpp"

#include "lvr2/registration/TransformUtils.hpp"
#include "lvr2/registration/ICPPointAlign.hpp"
#include "lvr2/util/Util.hpp"

#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkPointPicker.h>
#include <vtkPointData.h>
#include <vtkAreaPicker.h>
#include <vtkCamera.h>
#include <vtkDefaultPass.h>
#include <vtkCubeSource.h>

#include "../vtkBridge/LVRChunkedMeshBridge.hpp"
#include "../vtkBridge/LVRChunkedMeshCuller.hpp"


#include <QString>

#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>

#include <QMetaType>

#include "Options.hpp"

namespace lvr2
{

using Vec = BaseVector<float>;
const string LVRMainWindow::UNKNOWNNAME = "Unlabeled";

LVRMainWindow::LVRMainWindow()
{
    setupUi(this);
    setupQVTK();

    // Init members
    m_correspondanceDialog = new LVRCorrespondanceDialog(treeWidget);
    m_labelDialog = new LVRLabelDialog(treeWidget);
    m_incompatibilityBox = new QMessageBox();
    m_aboutDialog = new QDialog(this);
    Ui::AboutDialog aboutDialog;
    aboutDialog.setupUi(m_aboutDialog);

    m_errorDialog = new QDialog(this);
    Ui::TooltipDialog tooltipDialog;
    tooltipDialog.setupUi(m_errorDialog);

    m_previewPointBuffer = nullptr;

    // Setup specific properties
    QHeaderView* v = this->treeWidget->header();
    v->resizeSection(0, 175);

    treeWidget->setSelectionMode(QAbstractItemView::ExtendedSelection);
    treeWidget->setContextMenuPolicy(Qt::CustomContextMenu);

    m_treeWidgetHelper = new LVRTreeWidgetHelper(treeWidget);

    
    m_actionCopyModelItem = new QAction("Copy item", this);
    m_actionCopyModelItem->setShortcut(QKeySequence::Copy);
    m_actionCopyModelItem->setShortcutContext(Qt::ApplicationShortcut);
#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
    m_actionCopyModelItem->setShortcutVisibleInContextMenu(true);
#endif


    m_actionPasteModelItem = new QAction("Paste item", this);
    m_actionPasteModelItem->setShortcut(QKeySequence::Paste);
    m_actionPasteModelItem->setShortcutContext(Qt::ApplicationShortcut);
#if QT_VERSION >= QT_VERSION_CHECK(5, 10, 0)
    m_actionPasteModelItem->setShortcutVisibleInContextMenu(true);
#endif


    m_actionRenameModelItem = new QAction("Rename item", this);
    m_actionDeleteModelItem = new QAction("Delete item", this);
    m_actionExportModelTransformed = new QAction("Export item with transformation", this);
    m_actionShowColorDialog = new QAction("Select base color...", this);
    m_actionLoadPointCloudData = new QAction("load PointCloud", this);
    m_actionUnloadPointCloudData = new QAction("unload PointCloud", this);

    m_actionAddLabelClass = new QAction("Add label class", this);
    m_actionAddNewInstance = new QAction("Add new instance", this);
    m_actionRemoveInstance = new QAction("Remove instance", this);

    m_actionShowImage = new QAction("Show Image", this);
    m_actionSetViewToCamera = new QAction("Set view to camera", this);

    this->addAction(m_actionCopyModelItem);
    this->addAction(m_actionPasteModelItem);

    m_labelTreeParentItemContextMenu = new QMenu();
    m_labelTreeParentItemContextMenu->addAction(m_actionAddLabelClass);
    m_labelTreeParentItemContextMenu->addAction(m_actionAddNewInstance);
    m_labelTreeChildItemContextMenu = new QMenu();
    m_labelTreeChildItemContextMenu->addAction(m_actionRemoveInstance);

    m_treeParentItemContextMenu = new QMenu;
    m_treeParentItemContextMenu->addAction(m_actionRenameModelItem);
    m_treeParentItemContextMenu->addAction(m_actionDeleteModelItem);
    m_treeParentItemContextMenu->addAction(m_actionCopyModelItem);

    m_treeChildItemContextMenu = new QMenu;
    m_treeChildItemContextMenu->addAction(m_actionExportModelTransformed);
    m_treeChildItemContextMenu->addAction(m_actionShowColorDialog);
    m_treeChildItemContextMenu->addAction(m_actionDeleteModelItem);
    m_treeChildItemContextMenu->addAction(m_actionCopyModelItem);

    m_PointPreviewPlotter = this->plotter;
    this->dockWidgetSpectralSliderSettings->close();
    this->dockWidgetSpectralColorGradientSettings->close();
    this->dockWidgetPointPreview->close();
    this->dockWidgetLabel->close();
 
    // Toolbar item "File"
    m_actionOpen = this->actionOpen;
    m_actionOpenChunkedMesh = this->actionOpenChunkedMesh;
    m_actionExport = this->actionExport;
    m_actionQuit = this->actionQuit;
    // Toolbar item "Views"
    m_actionReset_Camera = this->actionReset_Camera;
    m_actionStore_Current_View = this->actionStore_Current_View;
    m_actionRecall_Stored_View = this->actionRecall_Stored_View;
    m_actionCameraPathTool = this->actionCameraPathTool;
    // Toolbar item "Reconstruction"
    m_actionEstimate_Normals = this->actionEstimate_Normals; // TODO: fix normal estimation
    m_actionMarching_Cubes = this->actionMarching_Cubes;
    m_actionPlanar_Marching_Cubes = this->actionPlanar_Marching_Cubes;
    m_actionExtended_Marching_Cubes = this->actionExtended_Marching_Cubes;
    m_actionCompute_Textures = this->actionCompute_Textures; // TODO: Compute textures
    m_actionMatch_Textures_from_Package = this->actionMatch_Textures_from_Package; // TODO: Match textures from package
    m_actionExtract_and_Rematch_Patterns = this->actionExtract_and_Rematch_Patterns; // TODO: Extract and rematch patterns
    // Toolbar item "Mesh Optimization"
    m_actionPlanar_Optimization = this->actionPlanar_Optimization;
    m_actionRemove_Artifacts = this->actionRemove_Artifacts;
    // Toolbar item "Filtering"
    m_actionRemove_Outliers = this->actionRemove_Outliers;
    m_actionMLS_Projection = this->actionMLS_Projection;
    // Toolbar item "Registration"
    m_actionICP_Using_Manual_Correspondance = this->actionICP_Using_Manual_Correspondance;
    m_actionICP_Using_Pose_Estimations = this->actionICP_Using_Pose_Estimations; // TODO: implement ICP registration
    m_actionGlobal_Relaxation = this->actionGlobal_Relaxation; // TODO: implement global relaxation
    // Toolbar item "Classification"
    m_actionSimple_Plane_Classification = this->actionSimple_Plane_Classification;
    m_actionFurniture_Recognition = this->actionFurniture_Recognition;

    //Toolbar item "Labeling"
    m_actionStart_labeling = this->actionLabeling_Start;
    m_actionStop_labeling = this->actionLabeling_Stop;
    m_actionExtract_labeling = this->actionLabeling_Export;

    // Toolbar item "About"
    // TODO: Replace "About"-QMenu with "About"-QAction
    m_menuAbout = this->menuAbout;
    // QToolbar below toolbar
    m_actionShow_Points = this->actionShow_Points;
    m_actionShow_Normals = this->actionShow_Normals;
    m_actionShow_Mesh = this->actionShow_Mesh;
    m_actionShow_Wireframe = this->actionShow_Wireframe;
    m_actionShowBackgroundSettings = this->actionShowBackgroundSettings;
    m_actionShowSpectralSlider = this->actionShow_SpectralSlider;
    m_actionShowSpectralColorGradient = this->actionShow_SpectralColorGradient;
    m_actionShowSpectralPointPreview = this->actionShow_SpectralPointPreview;
    m_actionShowSpectralHistogram = this->actionShow_SpectralHistogram;

    // Slider below tree widget
//    m_horizontalSliderPointSize = this->horizontalSliderPointSize;
//    m_horizontalSliderTransparency = this->horizontalSliderTransparency;
//    // Combo boxes
//    m_comboBoxGradient = this->comboBoxGradient; // TODO: implement gradients
//    m_comboBoxShading = this->comboBoxShading; // TODO: fix shading
    // Buttons below combo boxes
    m_buttonCameraPathTool = this->buttonCameraPathTool;
    m_buttonCreateMesh = this->buttonCreateMesh;
    m_buttonExportData = this->buttonExportData;
    m_buttonTransformModel = this->buttonTransformModel;

    // Spectral Settings
    m_spectralSliders[0] = this->horizontalSlider_Hyperspectral_red;
    m_spectralSliders[1] = this->horizontalSlider_Hyperspectral_green;
    m_spectralSliders[2] = this->horizontalSlider_Hyperspectral_blue;
    m_spectralCheckboxes[0] = this->checkBox_hred;
    m_spectralCheckboxes[1] = this->checkBox_hgreen;
    m_spectralCheckboxes[2] = this->checkBox_hblue;
    m_spectralLabels[0] = this->label_hred;
    m_spectralLabels[1] = this->label_hgreen;
    m_spectralLabels[2] = this->label_hblue;
    m_spectralLineEdits[0] = this->lineEdit_hred;
    m_spectralLineEdits[1] = this->lineEdit_hgreen;
    m_spectralLineEdits[2] = this->lineEdit_hblue;

    m_gradientSlider = this->sliderGradientWavelength;
    m_gradientLineEdit = this->lineEditGradientWavelength;


    //vtkSmartPointer<vtkAreaPicker> areaPicker = vtkSmartPointer<vtkAreaPicker>::New();
    //qvtkWidget->GetRenderWindow()->GetInteractor()->SetPicker(areaPicker);
    vtkSmartPointer<vtkPointPicker> pointPicker = vtkSmartPointer<vtkPointPicker>::New();
    qvtkWidget->GetRenderWindow()->GetInteractor()->SetPicker(pointPicker);


   // Widget to display the coordinate system
     m_axes = vtkSmartPointer<vtkAxesActor>::New();

     m_axesWidget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
     m_axesWidget->SetOutlineColor( 0.9300, 0.5700, 0.1300 );
     m_axesWidget->SetOrientationMarker( m_axes );
     m_axesWidget->SetInteractor( m_renderer->GetRenderWindow()->GetInteractor() );
     m_axesWidget->SetDefaultRenderer(m_renderer);
     m_axesWidget->SetViewport( 0.0, 0.0, 0.3, 0.3 );
     m_axesWidget->SetEnabled( 1 );
     m_axesWidget->InteractiveOff();

     // Disable action if EDL is not available
#ifndef LVR2_USE_VTK_GE_7_1
     actionRenderEDM->setEnabled(false);
#endif

    connectSignalsAndSlots();

}

LVRMainWindow::~LVRMainWindow()
{
//    this->qvtkWidget->GetRenderWindow()->RemoveRenderer(m_renderer);

    if(m_correspondanceDialog)
    {
        delete m_correspondanceDialog;
    }
    if(m_labelDialog)
    {
        delete m_labelDialog;
    }

    if (m_pickingInteractor)
    {
        qvtkWidget->GetRenderWindow()->GetInteractor()->SetInteractorStyle(nullptr);
        m_pickingInteractor->Delete();
    }

    if (m_treeParentItemContextMenu)
    {
        delete m_treeParentItemContextMenu;
    }

    if (m_treeChildItemContextMenu)
    {
        delete m_treeChildItemContextMenu;
    }
    if (m_treeWidgetHelper)
    {
        delete m_treeWidgetHelper;
    }

    if (m_aboutDialog)
    {
        delete m_aboutDialog;
    }
    if (m_errorDialog)
    {
        delete m_errorDialog;
    }
    delete m_incompatibilityBox;

    delete m_actionRenameModelItem;
    delete m_actionDeleteModelItem;
    delete m_actionCopyModelItem;
    delete m_actionPasteModelItem;
    delete m_actionExportModelTransformed;
    delete m_actionShowColorDialog;
    delete m_actionLoadPointCloudData;
    delete m_actionUnloadPointCloudData;
    delete m_actionShowImage;
    delete m_actionSetViewToCamera;
    
}

void LVRMainWindow::connectSignalsAndSlots()
{
    QObject::connect(m_actionOpen, SIGNAL(triggered()), this, SLOT(loadModel()));
    QObject::connect(m_actionOpenChunkedMesh, SIGNAL(triggered()), this, SLOT(loadChunkedMesh()));
    QObject::connect(m_actionExport, SIGNAL(triggered()), this, SLOT(exportSelectedModel()));
    QObject::connect(this->actionOpenLabeledPointcloud, SIGNAL(triggered()), this, SLOT(loadLabels()));
    QObject::connect(this->actionExportLabeledPointcloud, SIGNAL(triggered()), this, SLOT(exportLabels()));
    QObject::connect(treeWidget, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(showTreeContextMenu(const QPoint&)));
    QObject::connect(treeWidget, SIGNAL(itemSelectionChanged()), this, SLOT(restoreSliders()));
    QObject::connect(treeWidget, SIGNAL(itemSelectionChanged()), this, SLOT(highlightBoundingBoxes()));
    QObject::connect(treeWidget, SIGNAL(itemChanged(QTreeWidgetItem*, int)), this, SLOT(setModelVisibility(QTreeWidgetItem*, int)));

    QObject::connect(labelTreeWidget, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(showLabelTreeContextMenu(const QPoint&)));


    QObject::connect(m_actionQuit, SIGNAL(triggered()), qApp, SLOT(quit()));

    QObject::connect(m_actionShowColorDialog, SIGNAL(triggered()), this, SLOT(showColorDialog()));
    QObject::connect(m_actionRenameModelItem, SIGNAL(triggered()), this, SLOT(renameModelItem()));
    QObject::connect(m_actionDeleteModelItem, SIGNAL(triggered()), this, SLOT(deleteModelItem()));
    QObject::connect(m_actionCopyModelItem, SIGNAL(triggered()), this, SLOT(copyModelItem()));
    QObject::connect(m_actionPasteModelItem, SIGNAL(triggered()), this, SLOT(pasteModelItem()));
    QObject::connect(m_actionLoadPointCloudData, SIGNAL(triggered()), this, SLOT(loadPointCloudData()));
    QObject::connect(m_actionUnloadPointCloudData, SIGNAL(triggered()), this, SLOT(unloadPointCloudData()));

    QObject::connect(m_actionAddLabelClass, SIGNAL(triggered()), this, SLOT(addLabelClass()));

    QObject::connect(selectedInstanceComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(comboBoxIndexChanged(int)));

    QObject::connect(m_actionShowImage, SIGNAL(triggered()), this, SLOT(showImage()));
    QObject::connect(m_actionSetViewToCamera, SIGNAL(triggered()), this, SLOT(setViewToCamera()));


    QObject::connect(m_actionExportModelTransformed, SIGNAL(triggered()), this, SLOT(exportSelectedModel()));

    QObject::connect(m_actionReset_Camera, SIGNAL(triggered()), this, SLOT(updateView()));
    QObject::connect(m_actionStore_Current_View, SIGNAL(triggered()), this, SLOT(saveCamera()));
    QObject::connect(m_actionRecall_Stored_View, SIGNAL(triggered()), this, SLOT(loadCamera()));
    QObject::connect(m_actionCameraPathTool, SIGNAL(triggered()), this, SLOT(openCameraPathTool()));

    QObject::connect(m_actionEstimate_Normals, SIGNAL(triggered()), this, SLOT(estimateNormals()));
    QObject::connect(m_actionMarching_Cubes, SIGNAL(triggered()), this, SLOT(reconstructUsingMarchingCubes()));
    QObject::connect(m_actionPlanar_Marching_Cubes, SIGNAL(triggered()), this, SLOT(reconstructUsingPlanarMarchingCubes()));
    QObject::connect(m_actionExtended_Marching_Cubes, SIGNAL(triggered()), this, SLOT(reconstructUsingExtendedMarchingCubes()));

    QObject::connect(m_actionPlanar_Optimization, SIGNAL(triggered()), this, SLOT(optimizePlanes()));
    QObject::connect(m_actionRemove_Artifacts, SIGNAL(triggered()), this, SLOT(removeArtifacts()));

    QObject::connect(m_actionRemove_Outliers, SIGNAL(triggered()), this, SLOT(removeOutliers()));
    QObject::connect(m_actionMLS_Projection, SIGNAL(triggered()), this, SLOT(applyMLSProjection()));

    QObject::connect(m_actionICP_Using_Manual_Correspondance, SIGNAL(triggered()), this, SLOT(manualICP()));

    QObject::connect(m_menuAbout, SIGNAL(triggered(QAction*)), m_aboutDialog, SLOT(show()));

    QObject::connect(actionRenderEDM, SIGNAL(toggled(bool)), this, SLOT(toogleEDL(bool)));

    QObject::connect(m_actionShow_Points, SIGNAL(toggled(bool)), this, SLOT(togglePoints(bool)));
    QObject::connect(m_actionShow_Normals, SIGNAL(toggled(bool)), this, SLOT(toggleNormals(bool)));
    QObject::connect(m_actionShow_Mesh, SIGNAL(toggled(bool)), this, SLOT(toggleMeshes(bool)));
    QObject::connect(m_actionShow_Wireframe, SIGNAL(toggled(bool)), this, SLOT(toggleWireframe(bool)));
    QObject::connect(m_actionShowBackgroundSettings, SIGNAL(triggered()), this, SLOT(showBackgroundDialog()));
    QObject::connect(m_actionShowSpectralSlider, SIGNAL(triggered()), dockWidgetSpectralSliderSettings, SLOT(show()));
    QObject::connect(m_actionShowSpectralColorGradient, SIGNAL(triggered()), dockWidgetSpectralColorGradientSettings, SLOT(show()));
    QObject::connect(m_actionShowSpectralPointPreview, SIGNAL(triggered()), dockWidgetPointPreview, SLOT(show()));
    QObject::connect(m_actionShowSpectralHistogram, SIGNAL(triggered()), this, SLOT(showHistogramDialog()));

//    QObject::connect(m_horizontalSliderPointSize, SIGNAL(valueChanged(int)), this, SLOT(changePointSize(int)));
//    QObject::connect(m_horizontalSliderTransparency, SIGNAL(valueChanged(int)), this, SLOT(changeTransparency(int)));

//    QObject::connect(m_comboBoxShading, SIGNAL(currentIndexChanged(int)), this, SLOT(changeShading(int)));

    QObject::connect(m_buttonCameraPathTool, SIGNAL(pressed()), this, SLOT(openCameraPathTool()));
    QObject::connect(m_buttonCreateMesh, SIGNAL(pressed()), this, SLOT(reconstructUsingMarchingCubes()));
    QObject::connect(m_buttonExportData, SIGNAL(pressed()), this, SLOT(exportSelectedModel()));
    QObject::connect(m_buttonTransformModel, SIGNAL(pressed()), this, SLOT(showTransformationDialog()));

    for (int i = 0; i < 3; i++)
    {
        QObject::connect(m_spectralSliders[i], SIGNAL(valueChanged(int)), this, SLOT(onSpectralSliderChanged()));
        QObject::connect(m_spectralSliders[i], SIGNAL(actionTriggered(int)), this, SLOT(onSpectralSliderChanged(int)));
        QObject::connect(m_spectralSliders[i], SIGNAL(sliderReleased()), this, SLOT(changeSpectralColor()));
        QObject::connect(m_spectralCheckboxes[i], SIGNAL(stateChanged(int)), this, SLOT(changeSpectralColor()));
        QObject::connect(m_spectralLineEdits[i], SIGNAL(textChanged(QString)), this, SLOT(onSpectralLineEditChanged()));
        QObject::connect(m_spectralLineEdits[i], SIGNAL(editingFinished()), this, SLOT(onSpectralLineEditSubmit()));
    }

    QObject::connect(m_gradientLineEdit, SIGNAL(textChanged(QString)), this, SLOT(onGradientLineEditChanged()));
    QObject::connect(m_gradientLineEdit, SIGNAL(editingFinished()), this, SLOT(onGradientLineEditSubmit()));
    QObject::connect(m_gradientSlider, SIGNAL(valueChanged(int)), this, SLOT(onGradientSliderChanged()));
    QObject::connect(m_gradientSlider, SIGNAL(actionTriggered(int)), this, SLOT(onGradientSliderChanged(int)));
    QObject::connect(m_gradientSlider, SIGNAL(sliderReleased()), this, SLOT(changeGradientColor()));

    QObject::connect(comboBox_colorgradient, SIGNAL(currentIndexChanged(int)), this, SLOT(changeGradientColor()));
    QObject::connect(checkBox_normcolors, SIGNAL(stateChanged(int)), this, SLOT(changeGradientColor()));
    QObject::connect(checkBox_NDVI, SIGNAL(stateChanged(int)), this, SLOT(changeGradientColor()));

    QObject::connect(m_pickingInteractor, SIGNAL(firstPointPicked(double*)),m_correspondanceDialog, SLOT(firstPointPicked(double*)));
    QObject::connect(m_pickingInteractor, SIGNAL(secondPointPicked(double*)),m_correspondanceDialog, SLOT(secondPointPicked(double*)));
    QObject::connect(m_pickingInteractor, SIGNAL(pointSelected(vtkActor*, int)), this, SLOT(showPointPreview(vtkActor*, int)));
    QObject::connect(m_pickingInteractor, SIGNAL(pointsLabeled(uint16_t, int)), m_labelDialog, SLOT(updatePointCount(uint16_t, int)));
    QObject::connect(m_pickingInteractor, SIGNAL(pointsLabeled(const uint16_t, const int)), this, SLOT(updatePointCount(const uint16_t, const int)));
    QObject::connect(m_pickingInteractor, SIGNAL(responseLabels(std::vector<uint16_t>)), m_labelDialog, SLOT(responseLabels(std::vector<uint16_t>)));

    QObject::connect(this, SIGNAL(labelAdded(QTreeWidgetItem*)), m_pickingInteractor, SLOT(newLabel(QTreeWidgetItem*)));
    QObject::connect(m_labelDialog, SIGNAL(labelAdded(QTreeWidgetItem*)), m_pickingInteractor, SLOT(newLabel(QTreeWidgetItem*)));
    QObject::connect(m_labelDialog, SIGNAL(labelLoaded(int, std::vector<int>)), m_pickingInteractor, SLOT(setLabel(int, std::vector<int>)));
    QObject::connect(m_labelDialog->m_ui->exportLabelButton, SIGNAL(pressed()), m_pickingInteractor, SLOT(requestLabels()));
    QObject::connect(m_labelDialog, SIGNAL(hidePoints(int, bool)), m_pickingInteractor, SLOT(setLabeledPointVisibility(int, bool)));
    QObject::connect(this, SIGNAL(hidePoints(int, bool)), m_pickingInteractor, SLOT(setLabeledPointVisibility(int, bool)));

    QObject::connect(labelTreeWidget, SIGNAL(itemChanged(QTreeWidgetItem*, int)), this, SLOT(visibilityChanged(QTreeWidgetItem*, int)));
    QObject::connect(m_labelDialog, SIGNAL(labelChanged(uint16_t)), m_pickingInteractor, SLOT(labelSelected(uint16_t)));
    QObject::connect(this, SIGNAL(labelChanged(uint16_t)), m_pickingInteractor, SLOT(labelSelected(uint16_t)));

    QObject::connect(labelTreeWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(cellSelected(QTreeWidgetItem*, int)));
    QObject::connect(m_labelDialog->m_ui->lassotoolButton, SIGNAL(toggled(bool)), m_pickingInteractor, SLOT(setLassoTool(bool)));

    // Interaction with interactor
    QObject::connect(this->doubleSpinBoxDollySpeed, SIGNAL(valueChanged(double)), m_pickingInteractor, SLOT(setMotionFactor(double)));
    QObject::connect(this->doubleSpinBoxRotationSpeed, SIGNAL(valueChanged(double)), m_pickingInteractor, SLOT(setRotationFactor(double)));
    QObject::connect(this->checkBoxShowFocal, SIGNAL(stateChanged(int)), m_pickingInteractor, SLOT(setFocalPointRendering(int)));
    QObject::connect(this->checkBoxStereo, SIGNAL(stateChanged(int)), m_pickingInteractor, SLOT(setStereoMode(int)));
    QObject::connect(this->buttonPickFocal, SIGNAL(pressed()), m_pickingInteractor, SLOT(pickFocalPoint()));
    QObject::connect(this->pushButtonTerrain, SIGNAL(pressed()), m_pickingInteractor, SLOT(modeTerrain()));
    QObject::connect(this->buttonResetCamera, SIGNAL(pressed()), m_pickingInteractor, SLOT(resetCamera()));
    QObject::connect(this->pushButtonTrackball, SIGNAL(pressed()), m_pickingInteractor, SLOT(modeTrackball()));
    QObject::connect(this->pushButtonFly , SIGNAL(pressed()), m_pickingInteractor, SLOT(modeShooter()));


    QObject::connect(this->actionSelected_Lasso, SIGNAL(triggered()), this, SLOT(manualLabeling()));
    QObject::connect(this->actionSelected_Polygon, SIGNAL(triggered()), this, SLOT(manualLabeling()));
    QObject::connect(m_actionStop_labeling, SIGNAL(triggered()), this, SLOT(manualLabeling()));
//    QObject::connect(m_labelInteractor, SIGNAL(pointsSelected()), this, SLOT(manualLabeling()));
 //   QObject::connect(m_labelInteractor, SIGNAL(pointsSelected()), m_labelDialog, SLOT(labelPoints()));
//    QObject::connect(m_actionExtract_labeling, SIGNAL(triggered()), m_labelInteractor, SLOT(extractLabel()));
    QObject::connect(m_correspondanceDialog, SIGNAL(disableCorrespondenceSearch()), m_pickingInteractor, SLOT(correspondenceSearchOff()));
    QObject::connect(m_pickingInteractor, SIGNAL(labelingStarted(bool)), this, SLOT(changePicker(bool)));
    QObject::connect(m_correspondanceDialog, SIGNAL(enableCorrespondenceSearch()), m_pickingInteractor, SLOT(correspondenceSearchOn()));
    QObject::connect(m_correspondanceDialog->m_dialog, SIGNAL(accepted()), m_pickingInteractor, SLOT(correspondenceSearchOff()));
    QObject::connect(m_correspondanceDialog->m_dialog, SIGNAL(rejected()), m_pickingInteractor, SLOT(correspondenceSearchOff()));
    QObject::connect(m_correspondanceDialog->m_dialog, SIGNAL(accepted()), this, SLOT(alignPointClouds()));

    QObject::connect(m_correspondanceDialog, SIGNAL(addArrow(LVRVtkArrow*)), this, SLOT(addArrow(LVRVtkArrow*)));
    QObject::connect(m_correspondanceDialog, SIGNAL(removeArrow(LVRVtkArrow*)), this, SLOT(removeArrow(LVRVtkArrow*)));


    QObject::connect(plotter, SIGNAL(mouseRelease()), this, SLOT(showPointInfoDialog()));

    QObject::connect(radioButtonUseSpectralSlider, SIGNAL(toggled(bool)), this, SLOT(updateSpectralSlidersEnabled(bool)));
    QObject::connect(radioButtonUseSpectralGradient, SIGNAL(toggled(bool)), this, SLOT(updateSpectralGradientEnabled(bool)));

    QObject::connect(this, SIGNAL(correspondenceDialogOpened()), m_pickingInteractor, SLOT(correspondenceSearchOn()));
}

void LVRMainWindow::showBackgroundDialog()
{
    LVRBackgroundDialog dialog(qvtkWidget->GetRenderWindow());
    if(dialog.exec() == QDialog::Accepted)
    {
        if(dialog.renderGradient())
        {
            float r1, r2, g1, g2, b1, b2;
            dialog.getColor1(r1, g1, b1);
            dialog.getColor2(r2, g2, b2);
            m_renderer->GradientBackgroundOn();
            m_renderer->SetBackground(r1, g1, b1);
            m_renderer->SetBackground2(r2, g2, b2);
        }
        else
        {
            float r, g, b;
            dialog.getColor1(r, g, b);
            m_renderer->GradientBackgroundOff();
            m_renderer->SetBackground(r, g, b);
        }
        this->qvtkWidget->GetRenderWindow()->Render();
    }
}

void LVRMainWindow::setupQVTK()
{
#ifndef LVR2_USE_VTK8
    // z buffer fix
    QSurfaceFormat surfaceFormat = qvtkWidget->windowHandle()->format();
    surfaceFormat.setStencilBufferSize(8);
    qvtkWidget->windowHandle()->setFormat(surfaceFormat);
#endif

    // Grab relevant entities from the qvtk widget
    m_renderer = vtkSmartPointer<vtkRenderer>::New();

#ifdef LVR2_USE_VTK_GE_7_1
        m_renderer->TwoSidedLightingOn ();
        m_renderer->UseHiddenLineRemovalOff();
        m_renderer->RemoveAllLights();
#endif

    // Setup decent background colors
    m_renderer->GradientBackgroundOn();
    m_renderer->SetBackground(0.8, 0.8, 0.9);
    m_renderer->SetBackground2(1.0, 1.0, 1.0);

    vtkSmartPointer<vtkRenderWindow> renderWindow = this->qvtkWidget->GetRenderWindow();

    m_renderWindowInteractor = this->qvtkWidget->GetInteractor();
    m_renderWindowInteractor->Initialize();


    // Camera that saves a position that can be loaded
    m_camera = vtkSmartPointer<vtkCamera>::New();

    // Custom interactor to handle picking actions
    //m_pickingInteractor = new LVRPickingInteractor();
    //m_labelInteractor = LVRLabelInteractorStyle::New();
    m_pickingInteractor = LVRPickingInteractor::New();
    m_pickingInteractor->setRenderer(m_renderer);

    qvtkWidget->GetRenderWindow()->GetInteractor()->SetInteractorStyle( m_pickingInteractor );
   // qvtkWidget->GetRenderWindow()->GetInteractor()->SetInteractorStyle( m_labelInteractor );

    //vtkSmartPointer<vtkPointPicker> pointPicker = vtkSmartPointer<vtkPointPicker>::New();
    vtkSmartPointer<vtkAreaPicker> pointPicker = vtkSmartPointer<vtkAreaPicker>::New();
    qvtkWidget->GetRenderWindow()->GetInteractor()->SetPicker(pointPicker);

    // Camera and camera interpolator to be used for camera paths
    m_pathCamera = vtkSmartPointer<vtkCameraRepresentation>::New();
    vtkSmartPointer<vtkCameraInterpolator> cameraInterpolator = vtkSmartPointer<vtkCameraInterpolator>::New();
    cameraInterpolator->SetInterpolationTypeToSpline();
    m_pathCamera->SetInterpolator(cameraInterpolator);
    m_pathCamera->SetCamera(m_renderer->GetActiveCamera());


#ifdef LVR2_USE_VTK_GE_7_1
    // Enable EDL per default
    qvtkWidget->GetRenderWindow()->SetMultiSamples(0);

    m_basicPasses = vtkRenderStepsPass::New();
    m_edl = vtkEDLShading::New();
    m_edl->SetDelegatePass(m_basicPasses);
    vtkOpenGLRenderer *glrenderer = vtkOpenGLRenderer::SafeDownCast(m_renderer);

    glrenderer->SetPass(m_edl);
#endif

    // Finalize QVTK setup by adding the renderer to the window
    renderWindow->AddRenderer(m_renderer);

}

void LVRMainWindow::toogleEDL(bool state)
{
#ifdef LVR2_USE_VTK_GE_7_1
    vtkOpenGLRenderer *glrenderer = vtkOpenGLRenderer::SafeDownCast(m_renderer);

    if(state == false)
    {
        glrenderer->SetPass(m_basicPasses);
    }
    else
    {
        glrenderer->SetPass(m_edl);
    }
    this->qvtkWidget->GetRenderWindow()->Render();
#endif
}


void LVRMainWindow::updateView()
{
    m_renderer->ResetCamera();
    m_renderer->ResetCameraClippingRange();
    this->qvtkWidget->GetRenderWindow()->Render();

    // Estimate cam speed -> imagine a plausible number
    // of move operations to reach the focal point
    vtkCamera* cam = m_renderer->GetActiveCamera();
    double step = cam->GetDistance() / 100;

    this->doubleSpinBoxDollySpeed->setValue(step);

    // Signal that focal point of camera may have changed
    m_pickingInteractor->updateFocalPoint();
}

void LVRMainWindow::refreshView()
{
    this->qvtkWidget->GetRenderWindow()->Render();
}

void LVRMainWindow::saveCamera()
{
    m_camera->DeepCopy(m_renderer->GetActiveCamera());
}

void LVRMainWindow::loadCamera()
{
    m_renderer->GetActiveCamera()->DeepCopy(m_camera);
    refreshView();
}

void LVRMainWindow::openCameraPathTool()
{
    new LVRAnimationDialog(m_renderWindowInteractor, m_pathCamera, treeWidget);
}

void LVRMainWindow::addArrow(LVRVtkArrow* a)
{
    if(a)
    {
        m_renderer->AddActor(a->getArrowActor());
        m_renderer->AddActor(a->getStartActor());
        m_renderer->AddActor(a->getEndActor());
    }
    this->qvtkWidget->GetRenderWindow()->Render();
}

void LVRMainWindow::removeArrow(LVRVtkArrow* a)
{
    if(a)
    {
        m_renderer->RemoveActor(a->getArrowActor());
        m_renderer->RemoveActor(a->getStartActor());
        m_renderer->RemoveActor(a->getEndActor());
    }
    this->qvtkWidget->GetRenderWindow()->Render();
}

void LVRMainWindow::restoreSliders()
{
    std::set<LVRPointCloudItem*> pointCloudItems = getSelectedPointCloudItems();
    std::set<LVRMeshItem*> meshItems = getSelectedMeshItems();

    if (!pointCloudItems.empty())
    {
        LVRPointCloudItem* pointCloudItem = *pointCloudItems.begin();

//        m_horizontalSliderPointSize->setEnabled(true);
//        m_horizontalSliderPointSize->setValue(pointCloudItem->getPointSize());
        int transparency = ((float)1 - pointCloudItem->getOpacity()) * 100;
//        m_horizontalSliderTransparency->setEnabled(true);
//        m_horizontalSliderTransparency->setValue(transparency);

        color<size_t> channels;
        color<bool> use_channel;
        size_t n_channels, gradient_channel;
        bool use_ndvi, normalize_gradient;
        GradientType gradient_type;

        pointCloudItem->getPointBufferBridge()->getSpectralChannels(channels, use_channel);
        pointCloudItem->getPointBufferBridge()->getSpectralColorGradient(gradient_type, gradient_channel, normalize_gradient, use_ndvi);

        PointBufferPtr p = pointCloudItem->getPointBuffer();
        UCharChannelOptional spec_channels = p->getUCharChannel("spectral_channels");

        if (spec_channels)
        {
            n_channels = spec_channels->width();
            int wavelength_min = *p->getIntAtomic("spectral_wavelength_min");
            int wavelength_max = *p->getIntAtomic("spectral_wavelength_max");

            this->dockWidgetSpectralSliderSettingsContents->setEnabled(false); // disable to stop changeSpectralColor from re-rendering 6 times
            for (int i = 0; i < 3; i++)
            {
                m_spectralSliders[i]->setMaximum(wavelength_max - 1);
                m_spectralSliders[i]->setMinimum(wavelength_min);
                m_spectralSliders[i]->setSingleStep(Util::wavelengthPerChannel(p));
                m_spectralSliders[i]->setPageStep(10 * Util::wavelengthPerChannel(p));
                m_spectralSliders[i]->setValue(Util::getSpectralWavelength(channels[i], p));
                m_spectralSliders[i]->setEnabled(use_channel[i]);
                m_spectralLineEdits[i]->setEnabled(use_channel[i]);

                m_spectralCheckboxes[i]->setChecked(use_channel[i]);

                m_spectralLineEdits[i]->setText(QString("%1").arg(Util::getSpectralWavelength(channels[i], p)));
            }
            this->dockWidgetSpectralSliderSettingsContents->setEnabled(true);

            this->dockWidgetSpectralColorGradientSettingsContents->setEnabled(false);
            m_gradientSlider->setMaximum(wavelength_max - 1);
            m_gradientSlider->setMinimum(wavelength_min);
            m_gradientSlider->setValue(Util::getSpectralWavelength(gradient_channel, p));
            m_gradientSlider->setEnabled(!use_ndvi);
            m_gradientLineEdit->setEnabled(!use_ndvi);

            this->checkBox_NDVI->setChecked(use_ndvi);
            this->checkBox_normcolors->setChecked(normalize_gradient);
            this->comboBox_colorgradient->setCurrentIndex((int)gradient_type);
            m_gradientLineEdit->setText(QString("%1").arg(Util::getSpectralWavelength(gradient_channel, p)));
            this->dockWidgetSpectralColorGradientSettingsContents->setEnabled(true);
        }
        else
        {
            this->dockWidgetSpectralSliderSettingsContents->setEnabled(false);
            this->dockWidgetSpectralColorGradientSettingsContents->setEnabled(false);
        }
    }
    else
    {
//        m_horizontalSliderPointSize->setEnabled(false);
//        m_horizontalSliderPointSize->setValue(1);

        this->dockWidgetSpectralSliderSettingsContents->setEnabled(false);
        this->dockWidgetSpectralColorGradientSettingsContents->setEnabled(false);
    }

    if (!meshItems.empty())
    {
        LVRMeshItem* meshItem = *meshItems.begin();

        int transparency = ((float)1 - meshItem->getOpacity()) * 100;
//        m_horizontalSliderTransparency->setEnabled(true);
//        m_horizontalSliderTransparency->setValue(transparency);
    }

    if (pointCloudItems.empty() && meshItems.empty())
    {
//        m_horizontalSliderTransparency->setEnabled(false);
//        m_horizontalSliderTransparency->setValue(0);
    }
}

bool isSelfOrChildSelected(QTreeWidgetItem *item)
{

    bool selected = item->isSelected();

    for (int i = 0; i < item->childCount() && !selected; i++)
    {
        selected = isSelfOrChildSelected(item->child(i));
    }

    return selected;
}

void LVRMainWindow::highlightBoundingBoxes()
{
    QTreeWidgetItemIterator it(treeWidget);

    while (*it)
    {
        if ((*it)->type() == LVRBoundingBoxItemType)
        {
            LVRBoundingBoxItem *item = static_cast<LVRBoundingBoxItem *>(*it);
            item->getBoundingBoxBridge()->setColor(1.0, 1.0, 1.0);

            if (item->parent() && item->parent()->type() == LVRScanDataItemType)
            {
                QTreeWidgetItem *parent = item->parent();

                if (isSelfOrChildSelected(parent))
                {
                    item->getBoundingBoxBridge()->setColor(1.0, 1.0, 0.0);
                }
            }
        }

        if ((*it)->type() == LVRPointCloudItemType)
        {
            LVRPointCloudItem *item = static_cast<LVRPointCloudItem *>(*it);
            item->resetColor();

            QTreeWidgetItem *parent = item->parent();
            if (!parent || (parent->type() != LVRScanDataItemType && parent->type() != LVRModelItemType))
            {
                parent = *it;
            }

            if (isSelfOrChildSelected(parent))
            {
                QColor color;
                color.setRgbF(1.0, 1.0, 0.0);
                item->setSelectionColor(color);
            }
        }

        it++;
    }

    refreshView();
}

void LVRMainWindow::exportSelectedModel()
{
    // Get selected point cloud
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();
    if(items.size() > 0)
    {
        QTreeWidgetItem* item = items.first();
        if(item->type() == LVRPointCloudItemType)
        {
            if(item->parent() && item->parent()->type() == LVRModelItemType)
            {
                QString qFileName = QFileDialog::getSaveFileName(this, tr("Export Point Cloud As..."), "", tr("Point cloud Files(*.ply *.3d)"));

                LVRModelItem* model_item = static_cast<LVRModelItem*>(item->parent());
                LVRPointCloudItem* pc_item = static_cast<LVRPointCloudItem*>(item);
                PointBufferPtr points = pc_item->getPointBuffer();

                // Get transformation matrix
                Pose p = model_item->getPose();
                Matrix4<Vec> mat(Vec(p.x, p.y, p.z), Vec(p.r, p.t, p.p));

                // Allocate target buffer and insert transformed points
                size_t n = points->numPoints();
                floatArr transformedPoints(new float[3 * n]);
                floatArr pointArray = points->getPointArray();
                for(size_t i = 0; i < n; i++)
                {
                    Vec v(pointArray[3 * i], pointArray[3 * i + 1], pointArray[3 * i + 2]);
                    Vec vt = mat * v;

                    transformedPoints[3 * i    ] = vt[0];
                    transformedPoints[3 * i + 1] = vt[1];
                    transformedPoints[3 * i + 2] = vt[2];
                }

                // Save transformed points
                PointBufferPtr trans(new PointBuffer);
                trans->setPointArray(transformedPoints, n);
                ModelPtr model(new Model(trans));
                ModelFactory::saveModel(model, qFileName.toStdString());
            }
        }
    }
}

void LVRMainWindow::alignPointClouds()
{
    QString dataName = m_correspondanceDialog->getDataName();
    QString modelName = m_correspondanceDialog->getModelName();

    PointBufferPtr modelBuffer = m_treeWidgetHelper->getPointBuffer(modelName);
    PointBufferPtr dataBuffer  = m_treeWidgetHelper->getPointBuffer(dataName);

    LVRModelItem* dataItem = m_treeWidgetHelper->getModelItem(dataName);
    LVRModelItem* modelItem = m_treeWidgetHelper->getModelItem(modelName);
    if (!dataItem || !modelItem) {
        return;
    }

    Pose dataPose = dataItem->getPose();
    Eigen::Vector3f pos(dataPose.x, dataPose.y, dataPose.z);
    Eigen::Vector3f angles(dataPose.r, dataPose.t, dataPose.p);
    angles *= M_PI / 180.0; // degrees -> radians
    Transformf mat = poseToMatrix<float>(pos, angles);

    boost::optional<Transformf> correspondence = m_correspondanceDialog->getTransformation();
    if (correspondence.is_initialized())
    {
        mat *= correspondence.get();
        matrixToPose(mat, pos, angles);
        angles *= 180.0 / M_PI; // radians -> degrees

        dataItem->setPose(Pose {
            pos.x(), pos.y(), pos.z(),
            angles.x(), angles.y(), angles.z()
        });

        updateView();
    }

    // Refine pose via ICP
    if(m_correspondanceDialog->doICP() && modelBuffer && dataBuffer)
    {
        Pose modelPose = modelItem->getPose();
        pos = Eigen::Vector3f(modelPose.x, modelPose.y, modelPose.z);
        angles = Eigen::Vector3f(modelPose.r, modelPose.t, modelPose.p);
        angles /= 180.0 / M_PI;
        Transformf modelTransform = poseToMatrix<float>(pos, angles);

        /* TODO: convert to new ICPPointAlign

        ICPPointAlign icp(modelBuffer, dataBuffer, modelTransform, mat);
        icp.setEpsilon(m_correspondanceDialog->getEpsilon());
        icp.setMaxIterations(m_correspondanceDialog->getMaxIterations());
        icp.setMaxMatchDistance(m_correspondanceDialog->getMaxDistance());
        Matrix4d refinedTransform = icp.match();

        matrixToPose(refinedTransform, pos, angles);
        angles *= M_PI / 180.0; // radians -> degrees

        dataItem->setPose(Pose {
            pos.x(), pos.y(), pos.z(),
            angles.x(), angles.y(), angles.z()
        });
        */
    }
    m_correspondanceDialog->clearAllItems();
    updateView();
}

void LVRMainWindow::showLabelTreeContextMenu(const QPoint& p)
{
    QList<QTreeWidgetItem*> items = labelTreeWidget->selectedItems();
    QPoint globalPos = labelTreeWidget->mapToGlobal(p);
    if(items.size() > 0)
    {

        QTreeWidgetItem* item = items.first();
        if (item->type() == LVRLabelClassType)
        {
            auto selected = m_labelTreeParentItemContextMenu->exec(globalPos);
            if(selected == m_actionAddNewInstance)
            {
                addNewInstance(item);
            }
            return;
        }
        if(item->type() == LVRLabelInstanceType)
        {
            auto selected = m_labelTreeChildItemContextMenu->exec(globalPos);
            if(selected == m_actionRemoveInstance)
            {
                m_pickingInteractor->removeLabel(item->data(LABEL_ID_COLUMN, 0).toInt());
                auto topLevelItem = item->parent();
                //update the Count avoidign the standart "signal" case to avoid race conditions
                topLevelItem->setText(LABELED_POINT_COLUMN, QString::number(topLevelItem->text(LABELED_POINT_COLUMN).toInt() - item->text(LABELED_POINT_COLUMN).toInt()));
                topLevelItem->removeChild(item);
            }
            return;

        }
    }

    m_actionAddNewInstance->setEnabled(false);
    m_labelTreeParentItemContextMenu->exec(globalPos);
    m_actionAddNewInstance->setEnabled(true);
}

void LVRMainWindow::addLabelClass()
{
    //Ask For the Label name 
    bool accepted;
    QString className = QInputDialog::getText(this, tr("Choose name for new Label class"),
        tr("Class name:"), QLineEdit::Normal,
        tr("Labelname") , &accepted);
    if (!accepted || className.isEmpty())
    {
            //No valid Input
            return;
    }

    QColor label_color = QColorDialog::getColor(Qt::red, this, tr("Choose default Label Color for label Class(willbe used for first isntance)"));
    if (!label_color.isValid())
    {
            //Non Valid Color Return 
            return;
    }

    if (labelTreeWidget->topLevelItemCount() == 0)
    {

        //Setting up Top Label
        QTreeWidgetItem * item = new QTreeWidgetItem(LVRLabelClassType);
        item->setText(0, QString::fromStdString(UNKNOWNNAME));
        item->setText(LABELED_POINT_COLUMN, QString::number(0));
        item->setCheckState(LABEL_VISIBLE_COLUMN, Qt::Checked);
        item->setData(LABEL_ID_COLUMN, 1, QColor(Qt::red));
        //item->setFlags(item->flags() & ~ Qt::ItemIsSelectable);

        //Setting up new child item
        QTreeWidgetItem * childItem = new QTreeWidgetItem(LVRLabelInstanceType);
        childItem->setText(LABEL_NAME_COLUMN, QString::fromStdString(UNKNOWNNAME) + QString::number(1));
        childItem->setText(LABELED_POINT_COLUMN, QString::number(0));
        childItem->setCheckState(LABEL_VISIBLE_COLUMN, Qt::Checked);
        childItem->setData(LABEL_ID_COLUMN, 1, QColor(Qt::red));
        childItem->setData(LABEL_ID_COLUMN, 0, 0);
        item->addChild(childItem);
        labelTreeWidget->addTopLevelItem(item);    
        //Added first Top Level item enable instance button
        Q_EMIT(labelAdded(childItem));
        selectedInstanceComboBox->addItem(childItem->text(LABEL_NAME_COLUMN), 0);
    }

    int id = m_id++;
    //Setting up new Toplevel item
    QTreeWidgetItem * item = new QTreeWidgetItem(LVRLabelClassType);
    item->setText(0, className);
    item->setText(LABELED_POINT_COLUMN, QString::number(0));
    item->setCheckState(LABEL_VISIBLE_COLUMN, Qt::Checked);
    item->setData(LABEL_ID_COLUMN, 1, label_color);
    //item->setFlags(item->flags() & ~ Qt::ItemIsSelectable);

    //Setting up new child item
    QTreeWidgetItem * childItem = new QTreeWidgetItem(LVRLabelInstanceType);
    childItem->setText(LABEL_NAME_COLUMN, className + QString::number(1));
    childItem->setText(LABELED_POINT_COLUMN, QString::number(0));
    childItem->setCheckState(LABEL_VISIBLE_COLUMN, Qt::Checked);
    childItem->setData(LABEL_ID_COLUMN, 1, label_color);
    childItem->setData(LABEL_ID_COLUMN, 0, id);
    item->addChild(childItem);
    m_selectedLabelItem = childItem;
    labelTreeWidget->addTopLevelItem(item);    
    
    //Add label to combo box 
    selectedInstanceComboBox->addItem(childItem->text(LABEL_NAME_COLUMN), id);
    Q_EMIT(labelAdded(childItem));
}
void LVRMainWindow::showTreeContextMenu(const QPoint& p)
{
    // Only display context menu for point clounds and meshes
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();
    if(items.size() > 0)
    {
        QTreeWidgetItem* item = items.first();
        if(item->type() == LVRModelItemType)
        {
            QPoint globalPos = treeWidget->mapToGlobal(p);
            m_treeParentItemContextMenu->exec(globalPos);
        }
        if(item->type() == LVRPointCloudItemType || item->type() == LVRMeshItemType)
        {
            QPoint globalPos = treeWidget->mapToGlobal(p);
            m_treeChildItemContextMenu->exec(globalPos);
        }
        if (item->type() == LVRScanDataItemType)
        {
            QPoint globalPos = treeWidget->mapToGlobal(p);

            LVRScanDataItem *sdi = static_cast<LVRScanDataItem *>(item);
            QMenu *con_menu = new QMenu;

            if (sdi->isPointCloudLoaded())
            {
                con_menu->addAction(m_actionUnloadPointCloudData);
            }
            else
            {
                con_menu->addAction(m_actionLoadPointCloudData);
            }

            con_menu->addAction(m_actionDeleteModelItem);
            con_menu->addAction(m_actionCopyModelItem);
            if(m_items_copied.size() > 0)
            {
                con_menu->addAction(m_actionPasteModelItem);
            } 
            con_menu->exec(globalPos);

            delete con_menu;
        }
        if(item->type() == LVRCvImageItemType)
        {
            QPoint globalPos = treeWidget->mapToGlobal(p);
            QMenu *con_menu = new QMenu;

            LVRCvImageItem *cvi = static_cast<LVRCvImageItem *>(item);

            con_menu->addAction(m_actionShowImage);
            con_menu->exec(globalPos);

            delete con_menu;
        }
        if(item->type() == LVRCamDataItemType)
        {
            QPoint globalPos = treeWidget->mapToGlobal(p);
            QMenu *con_menu = new QMenu;

            LVRCamDataItem* cam = static_cast<LVRCamDataItem *>(item);

            con_menu->addAction(m_actionSetViewToCamera);
            con_menu->exec(globalPos);

            delete con_menu;
        }
    }
}

void LVRMainWindow::renameModelItem()
{
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();
    if(items.size() > 0)
    {
        QTreeWidgetItem* item = items.first();
        LVRModelItem* model_item = getModelItem(item);
        if(model_item != NULL) new LVRRenameDialog(model_item, treeWidget);
    }
}

LVRModelItem* LVRMainWindow::loadModelItem(QString name)
{
    std::cout << "model loaded" << std::endl;
    // Load model and generate vtk representation
    ModelPtr model = ModelFactory::readModel(name.toStdString());
    ModelBridgePtr bridge(new LVRModelBridge(model));
    bridge->addActors(m_renderer);

    // Add item for this model to tree widget
    QFileInfo info(name);
    QString base = info.fileName();
    LVRModelItem* item = new LVRModelItem(bridge, base);
    this->treeWidget->addTopLevelItem(item);
    item->setExpanded(true);

    // Read Pose file
    boost::filesystem::path poseFile = name.toStdString();

    for (auto& extension : { "pose", "dat", "frames" })
    {
        poseFile.replace_extension(extension);
        if (boost::filesystem::exists(poseFile))
        {
            cout << "Found Pose file: " << poseFile << endl;
            Transformf mat = getTransformationFromFile<float>(poseFile);
            BaseVector<float> pos, angles;
            getPoseFromMatrix<float>(pos, angles, mat.transpose());

            angles *= 180.0 / M_PI; // radians -> degrees

            item->setPose(Pose {
                pos.x, pos.y, pos.z,
                angles.x, angles.y, angles.z
            });

            break;
        }
    }
    return item;
}

void LVRMainWindow::loadChunkedMesh(const QStringList& filenames, std::vector<std::string> layers, int cacheSize, float highResDistance)
{
    if(filenames.size() > 0)
    {
        QTreeWidgetItem* lastItem = nullptr;

        QStringList::const_iterator it = filenames.begin();
        while(it != filenames.end())
        {
            QFileInfo info((*it));
            QString base = info.fileName();

            std::cout << base.toStdString() << std::endl;

            if (info.suffix() == "h5")
            {
//                std::vector<std::string> layers = {"mesh0", "mesh1"};
                std::cout << info.absoluteFilePath().toStdString() << std::endl;
                //m_chunkBridge =  std::make_unique<LVRChunkedMeshBridge>(info.absoluteFilePath().toStdString(), m_renderer, layers, cacheSize);
                m_chunkBridge = ChunkedMeshBridgePtr(new LVRChunkedMeshBridge(info.absoluteFilePath().toStdString(), m_renderer, layers, cacheSize));
                m_chunkBridge->addInitialActors(m_renderer);
                m_chunkCuller = new ChunkedMeshCuller(m_chunkBridge.get(), highResDistance);
                m_renderer->AddCuller(m_chunkCuller);
                qRegisterMetaType<actorMap > ("actorMap");
                QObject::connect(m_chunkBridge.get(), 
                        SIGNAL(updateHighRes(actorMap, actorMap)),
                        this,
                        SLOT(updateDisplayLists(actorMap, actorMap)),
                        Qt::QueuedConnection);
            }
            ++it;
        }
    }
}


void LVRMainWindow::loadModels(const QStringList& filenames)
{
    if(filenames.size() > 0)
    {
        QTreeWidgetItem* lastItem = nullptr;

        QStringList::const_iterator it = filenames.begin();
        while(it != filenames.end())
        {
            // check for h5
            QFileInfo info((*it));
            QString base = info.fileName();

            if (info.suffix() == "h5")
            {
                // h5 special loading case
                // special case h5:
                // scan data is stored as 
                QTreeWidgetItem *root = new QTreeWidgetItem(treeWidget);
                root->setText(0, base);

                QIcon icon;
                icon.addFile(QString::fromUtf8(":/qv_scandata_tree_icon.png"), QSize(), QIcon::Normal, QIcon::Off);
                root->setIcon(0, icon);

                std::shared_ptr<ScanDataManager> sdm(new ScanDataManager(info.absoluteFilePath().toStdString()));

                lastItem = addScans(sdm, root);

                root->setExpanded(true);

                // load mesh only
                ModelPtr model_ptr(new Model());
                std::shared_ptr<HDF5IO> h5_io_ptr(new HDF5IO(info.absoluteFilePath().toStdString()));
                if(h5_io_ptr->readMesh(model_ptr))
                {
                    ModelBridgePtr bridge(new LVRModelBridge(model_ptr));
                    bridge->addActors(m_renderer);

                    // Add item for this model to tree widget
                    LVRModelItem* item = new LVRModelItem(bridge, "mesh");
                    root->addChild(item);
                    item->setExpanded(false);
                    lastItem = item;
                }

            } else {
                lastItem = loadModelItem(*it);
            }

            ++it;
        }

        if (lastItem != nullptr)
        {
            for(QTreeWidgetItem* selected : treeWidget->selectedItems())
            {
                selected->setSelected(false);
            }
            lastItem->setSelected(true);
        }

        highlightBoundingBoxes();
        restoreSliders();
        assertToggles();
        updateView();


	vtkSmartPointer<vtkPoints> points = 
		vtkSmartPointer<vtkPoints>::New();
	QTreeWidgetItemIterator itu(treeWidget);
	LVRPointCloudItem* citem;
	
	while (*itu)
	{
        	QTreeWidgetItem* item = *itu;

		if ( item->type() == LVRPointCloudItemType)
		{
			citem = static_cast<LVRPointCloudItem*>(*itu);
			points->SetData(citem->getPointBufferBridge()->getPointCloudActor()->GetMapper()->GetInput()->GetPointData()->GetScalars());

			m_pickingInteractor->setPoints(citem->getPointBufferBridge()->getPolyIDData());
                        m_labelDialog->setPoints(item->parent()->text(0).toStdString(), citem->getPointBufferBridge()->getPolyData());
		}
		itu++;
	}

    }
}

void LVRMainWindow::loadModel()
{
    QStringList filenames = QFileDialog::getOpenFileNames(this, tr("Open Model"), "", tr("Model Files (*.ply *.obj *.pts *.3d *.txt *.h5)"));
    loadModels(filenames);
    
}

void LVRMainWindow::loadChunkedMesh()
{
    QStringList filenames = QFileDialog::getOpenFileNames(this, tr("Open chunked mesh"), "", tr("Chunked meshes (*.h5)"));
     bool ok;
     QString text = QInputDialog::getText(0, "Enter layers",
                                         "Layers whitspace seperated:", QLineEdit::Normal,
                                         "", &ok);
     std::vector<std::string> layers;
     std::string unseperated = text.toStdString();
     if(text.isEmpty())
     {
        layers = {"mesh0"};
     }
     else
     {
     
         boost::tokenizer<boost::char_separator<char>> tokens(unseperated, boost::char_separator<char>());
         layers = std::vector<std::string>(tokens.begin(), tokens.end());
     }


     std::cout << "LAYERS " ;
     for(auto &layer : layers)
     {
         std::cout << layer << " ";
     }

     std::cout << std::endl;
     double highResDistance = QInputDialog::getDouble(0, "highResDistance", "highResDistance");
    
     if(highResDistance < 0)
     {
        highResDistance = 0.0;
     }


     int cacheSize = QInputDialog::getInt(0, "cacheSize", "cache size");
    
     if(cacheSize < 0)
     {
        cacheSize = 0;
     }
    loadChunkedMesh(filenames, layers, cacheSize, highResDistance);
    
}

void LVRMainWindow::loadPointCloudData()
{
    std::cout << "loaded points" << std::endl;
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();
    if(items.size() > 0)
    {
        QTreeWidgetItem* item = items.first();

        if(item->type() == LVRScanDataItemType)
        {
            LVRScanDataItem *sd = static_cast<LVRScanDataItem *>(item);


            if (!sd->isPointCloudLoaded())
            {
                sd->loadPointCloudData(m_renderer);
                sd->setVisibility(true, m_actionShow_Points->isChecked());

                highlightBoundingBoxes();
                assertToggles();
                restoreSliders();
                refreshView();
            }
        }
    }

}

void LVRMainWindow::unloadPointCloudData()
{
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();
    if(items.size() > 0)
    {
        QTreeWidgetItem* item = items.first();

        if(item->type() == LVRScanDataItemType)
        {
            LVRScanDataItem *sd = static_cast<LVRScanDataItem *>(item);

            if (sd->isPointCloudLoaded())
            {
                sd->unloadPointCloudData(m_renderer);

                highlightBoundingBoxes();
                refreshView();
                restoreSliders();
                assertToggles();
            }
        }
    }

}

void LVRMainWindow::showImage()
{
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();

    if(items.size() > 0)
    {
        QTreeWidgetItem* item = items.first();

        if(item->type() == LVRCvImageItemType)
        {
            LVRCvImageItem *cvi = static_cast<LVRCvImageItem *>(item);

            cvi->openWindow();
        }
    }
}

void LVRMainWindow::setViewToCamera()
{
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();

    if(items.size() > 0)
    {
        QTreeWidgetItem* item = items.first();

        if(item->type() == LVRCamDataItemType)
        {
            LVRCamDataItem *cam = static_cast<LVRCamDataItem *>(item);

            cam->setCameraView();

            refreshView();
        }
    }
}

void LVRMainWindow::deleteModelItem()
{
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();
    if(items.size() > 0)
    {
        QTreeWidgetItem* item = items.first();

        if(item->type() == LVRModelItemType)
        {
            QTreeWidgetItemIterator it(item);

            while(*it)
            {
                QTreeWidgetItem* child_item = *it;
                if(child_item->type() == LVRPointCloudItemType && child_item->parent() == item)
                {
                    LVRPointCloudItem* pc_item = getPointCloudItem(item);
                    if(pc_item != NULL)
                    {
                        m_renderer->RemoveActor(pc_item->getActor());
                        if (m_histograms.count(pc_item))
                        {
                            m_histograms.erase(pc_item);
                        }
                    }
                }
                else if(child_item->type() == LVRMeshItemType && child_item->parent() == item)
                {
                    LVRMeshItem* mesh_item = getMeshItem(item);
                    if(mesh_item != NULL)
                    {
                        m_renderer->RemoveActor(mesh_item->getWireframeActor());
                        m_renderer->RemoveActor(mesh_item->getActor());
                    }
                }

                ++it;
            }
        }
        else
        {
            // Remove model from view
            LVRPointCloudItem* pc_item = getPointCloudItem(item);
            if(pc_item != NULL)
            {
                m_renderer->RemoveActor(pc_item->getActor());
                if (m_histograms.count(pc_item))
                {
                    m_histograms.erase(pc_item);
                }
            }

            LVRMeshItem* mesh_item = getMeshItem(item);
            if(mesh_item != NULL) m_renderer->RemoveActor(mesh_item->getActor());
        }

        // Remove list item (safe according to http://stackoverflow.com/a/9399167)
        delete item;

        refreshView();
        restoreSliders();
    }
}


void LVRMainWindow::copyModelItem()
{
    // std::cout << "COPY!" << std::endl;

    if(m_items_copied.size() == 0)
    {
        m_treeParentItemContextMenu->addAction(m_actionPasteModelItem);
        m_treeChildItemContextMenu->addAction(m_actionPasteModelItem);
    }

    m_items_copied = treeWidget->selectedItems();
}

void LVRMainWindow::pasteModelItem()
{

    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();

    if(items.size() > 0)
    {
        QTreeWidgetItem* to_item = items.first();

        for(QTreeWidgetItem* from_item : m_items_copied)
        {
            std::cout << "copy " << from_item->text(0).toStdString() << std::endl;
            QString name = from_item->text(0);

            // check if name already exist
            bool child_name_exists = false;
            bool recheck = true;

            while(childNameExists(to_item, name))
            {
                
                // TODO better
                name = increaseFilename(name);
                std::cout << "Change name to " << name.toStdString() << std::endl; 

            }

            QTreeWidgetItem* insert_item = from_item->clone();
            insert_item->setText(0, name);
            insert_item->setToolTip(0, name);

            // addChild removes all other childs?

            to_item->addChild(insert_item);

        }

        m_items_copied.clear();

        m_treeParentItemContextMenu->removeAction(m_actionPasteModelItem);
        m_treeChildItemContextMenu->removeAction(m_actionPasteModelItem);

    }

}

bool LVRMainWindow::childNameExists(QTreeWidgetItem* item, const QString& name)
{
    bool child_name_exists = false;

    const int num_children = item->childCount();

    for(int i=0; i<num_children; i++)
    {
        const QTreeWidgetItem* child = item->child(i);
        const QString child_name = child->text(0);
        if(name == child_name)
        {
            child_name_exists = true;
            break;
        }
    }

    return child_name_exists;
}

QString LVRMainWindow::increaseFilename(QString filename)
{
    QRegExp rx("(\\d+)$");
    
    if(rx.indexIn(filename, 0) != -1)
    {
        int number = 0;
        number = rx.cap(1).toInt();
        number += 1;
        filename.replace(rx, QString::number(number));
    } else {
        filename += "_1";
    }

    return filename;
}


LVRModelItem* LVRMainWindow::getModelItem(QTreeWidgetItem* item)
{
    if(item->type() == LVRModelItemType)
        return static_cast<LVRModelItem*>(item);

    if(item->parent() && item->parent()->type() == LVRModelItemType)
        return static_cast<LVRModelItem*>(item->parent());

    return NULL;
}

QList<LVRPointCloudItem*> LVRMainWindow::getPointCloudItems(QList<QTreeWidgetItem*> items)
{
    QList<LVRPointCloudItem*> pcs;

    for(QTreeWidgetItem* item : items)
    {
        if(item->type() == LVRPointCloudItemType)
        {
            pcs.append(static_cast<LVRPointCloudItem*>(item));
        } else if(item->type() == LVRModelItemType) {
            // get pc of model
            QTreeWidgetItemIterator it(item);
            while(*it)
            {
                QTreeWidgetItem* child_item = *it;
                if(child_item->type() == LVRPointCloudItemType
                    && child_item->parent() == item)
                {
                    pcs.append(static_cast<LVRPointCloudItem*>(child_item));
                }
                ++it;
            }

        } else if(item->type() == LVRScanDataItemType) {
            // Scan data selected: fetch pointcloud (transformed?)
            QTreeWidgetItemIterator it(item);
            while(*it)
            {
                QTreeWidgetItem* child_item = *it;
                if(child_item->type() == LVRPointCloudItemType
                    && child_item->parent() == item)
                {
                    // pointcloud found!
                    pcs.append(static_cast<LVRPointCloudItem*>(child_item));
                }

                ++it;
            }

        }

    }

    return pcs;
}

LVRPointCloudItem* LVRMainWindow::getPointCloudItem(QTreeWidgetItem* item)
{
    if(item->type() == LVRPointCloudItemType) return static_cast<LVRPointCloudItem*>(item);
    if(item->type() == LVRModelItemType)
    {
        QTreeWidgetItemIterator it(item);

        while(*it)
        {
            QTreeWidgetItem* child_item = *it;
            if(child_item->type() == LVRPointCloudItemType && child_item->parent() == item)
            {
                return static_cast<LVRPointCloudItem*>(child_item);
            }
            ++it;
        }
    }
    return NULL;
}

LVRMeshItem* LVRMainWindow::getMeshItem(QTreeWidgetItem* item)
{
    if(item->type() == LVRMeshItemType) return static_cast<LVRMeshItem*>(item);
    if(item->type() == LVRModelItemType)
    {
        QTreeWidgetItemIterator it(item);

        while(*it)
        {
            QTreeWidgetItem* child_item = *it;
            if(child_item->type() == LVRMeshItemType && child_item->parent() == item)
            {
                return static_cast<LVRMeshItem*>(child_item);
            }
            ++it;
        }
    }
    return NULL;
}

std::set<LVRModelItem*> LVRMainWindow::getSelectedModelItems()
{
    std::set<LVRModelItem*> items;
    for (QTreeWidgetItem* item : treeWidget->selectedItems())
    {
        LVRModelItem* modelItem = getModelItem(item);
        if (modelItem)
        {
            items.insert(modelItem);
        }
    }
    return items;
}
std::set<LVRPointCloudItem*> LVRMainWindow::getSelectedPointCloudItems()
{
    std::set<LVRPointCloudItem*> items;
    for (QTreeWidgetItem* item : treeWidget->selectedItems())
    {
        LVRPointCloudItem* pointCloudItem = getPointCloudItem(item);
        if (pointCloudItem)
        {
            items.insert(pointCloudItem);
        }
    }
    return items;
}
std::set<LVRMeshItem*> LVRMainWindow::getSelectedMeshItems()
{
    std::set<LVRMeshItem*> items;
    for (QTreeWidgetItem* item : treeWidget->selectedItems())
    {
        LVRMeshItem* modelItem = getMeshItem(item);
        if (modelItem)
        {
            items.insert(modelItem);
        }
    }
    return items;
}

void LVRMainWindow::assertToggles()
{
    togglePoints(m_actionShow_Points->isChecked());
    toggleNormals(m_actionShow_Normals->isChecked());
    toggleMeshes(m_actionShow_Mesh->isChecked());
    toggleWireframe(m_actionShow_Wireframe->isChecked());
}

void LVRMainWindow::setModelVisibility(QTreeWidgetItem* treeWidgetItem, int column)
{
    if(treeWidgetItem->type() == LVRModelItemType)
    {
        QTreeWidgetItemIterator it(treeWidgetItem);

        while(*it)
        {
            QTreeWidgetItem* child_item = *it;
            if(child_item->type() == LVRPointCloudItemType)
            {
                LVRModelItem* model_item = static_cast<LVRModelItem*>(treeWidgetItem);
                model_item->setModelVisibility(column, m_actionShow_Points->isChecked());
            }
            if(child_item->type() == LVRMeshItemType)
            {
                LVRModelItem* model_item = static_cast<LVRModelItem*>(treeWidgetItem);
                model_item->setModelVisibility(column, m_actionShow_Mesh->isChecked());
            }
            ++it;
        }

        refreshView();
    }
    else if (treeWidgetItem->type() == LVRScanDataItemType)
    {
        LVRScanDataItem *item = static_cast<LVRScanDataItem *>(treeWidgetItem);
        item->setVisibility(true, m_actionShow_Points->isChecked());

        refreshView();
    }
    else if (treeWidgetItem->type() == LVRCamDataItemType)
    {
        LVRCamDataItem *item = static_cast<LVRCamDataItem *>(treeWidgetItem);
        item->setVisibility(true);

        refreshView();
    }
    else if (treeWidgetItem->type() == LVRBoundingBoxItemType)
    {
        LVRBoundingBoxItem *item = static_cast<LVRBoundingBoxItem *>(treeWidgetItem);
        item->setVisibility(true);

        refreshView();
    }
    else if (treeWidgetItem->parent() && treeWidgetItem->parent()->type() == LVRScanDataItemType)
    {
        setModelVisibility(treeWidgetItem->parent(), column);
    }
}



void LVRMainWindow::changePointSize(int pointSize)
{
    for (QTreeWidgetItem* item : treeWidget->selectedItems())
    {
        if(item->type() == LVRModelItemType)
        {
            QTreeWidgetItemIterator it(item);

            while(*it)
            {
                QTreeWidgetItem* child_item = *it;
                if(child_item->type() == LVRPointCloudItemType && child_item->parent()->isSelected())
                {
                    LVRPointCloudItem* model_item = static_cast<LVRPointCloudItem*>(child_item);
                    model_item->setPointSize(pointSize);
                }
                ++it;
            }
        }
        else if(item->type() == LVRPointCloudItemType)
        {
            LVRPointCloudItem* model_item = static_cast<LVRPointCloudItem*>(item);
            model_item->setPointSize(pointSize);
        }

        refreshView();
    }
}

void LVRMainWindow::changeTransparency(int transparencyValue)
{
    for (QTreeWidgetItem* item : treeWidget->selectedItems())
    {
        float opacityValue = 1 - ((float)transparencyValue / (float)100);

        if(item->type() == LVRModelItemType)
        {
            QTreeWidgetItemIterator it(item);

            while(*it)
            {
                QTreeWidgetItem* child_item = *it;
                if(child_item->type() == LVRPointCloudItemType && child_item->parent()->isSelected())
                {
                    LVRPointCloudItem* model_item = static_cast<LVRPointCloudItem*>(child_item);
                    model_item->setOpacity(opacityValue);
                }
                else if(child_item->type() == LVRMeshItemType && child_item->parent()->isSelected())
                {
                    LVRMeshItem* model_item = static_cast<LVRMeshItem*>(child_item);
                    model_item->setOpacity(opacityValue);
                }
                ++it;
            }
        }
        else if(item->type() == LVRPointCloudItemType)
        {
            LVRPointCloudItem* model_item = static_cast<LVRPointCloudItem*>(item);
            model_item->setOpacity(opacityValue);
        }
        else if(item->type() == LVRMeshItemType)
        {
            LVRMeshItem* model_item = static_cast<LVRMeshItem*>(item);
            model_item->setOpacity(opacityValue);
        }

        refreshView();
    }
}

void LVRMainWindow::changeShading(int shader)
{
    for (QTreeWidgetItem* item : treeWidget->selectedItems())
    {
        if(item->type() == LVRMeshItemType)
        {
            LVRMeshItem* model_item = static_cast<LVRMeshItem*>(item);
            model_item->setShading(shader);
            refreshView();
        }
    }
}

void LVRMainWindow::togglePoints(bool checkboxState)
{
    QTreeWidgetItemIterator it(treeWidget);

    while(*it)
    {
        QTreeWidgetItem* item = *it;
        if(item->type() == LVRPointCloudItemType)
        {
            if (item->parent()->type() == LVRModelItemType)
            {
                LVRModelItem* model_item = static_cast<LVRModelItem*>(item->parent());
                if(model_item->isEnabled()) model_item->setVisibility(checkboxState);
            }
            if (item->parent()->type() == LVRScanDataItemType)
            {
                LVRScanDataItem* sd_item = static_cast<LVRScanDataItem*>(item->parent());
                sd_item->setVisibility(true, checkboxState);
            }
        }
        ++it;
    }

    refreshView();
}

void LVRMainWindow::toggleNormals(bool checkboxState)
{
    QTreeWidgetItemIterator it(treeWidget);

    while(*it)
    {
        QTreeWidgetItem* item = *it;
        if(item->type() == LVRPointCloudItemType)
        {
            if (item->parent()->type() == LVRModelItemType)
            {
                LVRModelItem* model_item = static_cast<LVRModelItem*>(item->parent());
                if(model_item->isEnabled()){
                    model_item->getModelBridge()->setNormalsVisibility(checkboxState);
                }
            }
        }
        ++it;
    }

    refreshView();
}

void LVRMainWindow::toggleMeshes(bool checkboxState)
{
    QTreeWidgetItemIterator it(treeWidget);

    while(*it)
    {
        QTreeWidgetItem* item = *it;
        if(item->type() == LVRMeshItemType)
        {
            LVRModelItem* model_item = static_cast<LVRModelItem*>(item->parent());
            if(model_item->isEnabled()) model_item->setVisibility(checkboxState);
        }
        ++it;
    }

    refreshView();
}

void LVRMainWindow::toggleWireframe(bool checkboxState)
{
    if(m_actionShow_Mesh)
    {
        QTreeWidgetItemIterator it(treeWidget);

        while(*it)
        {
            QTreeWidgetItem* item = *it;
            if(item->type() == LVRMeshItemType)
            {
                LVRMeshItem* mesh_item = static_cast<LVRMeshItem*>(item);
                if(checkboxState)
                {
                    m_renderer->AddActor(mesh_item->getWireframeActor());
                }
                else
                {
                    m_renderer->RemoveActor(mesh_item->getWireframeActor());
                }
                refreshView();
            }
            ++it;
        }

        refreshView();
    }
}

QTreeWidgetItem* LVRMainWindow::addScans(std::shared_ptr<ScanDataManager> sdm, QTreeWidgetItem *parent)
{
    QTreeWidgetItem *lastItem = nullptr;
    std::vector<ScanPtr> scans = sdm->getScans();
    std::vector<std::vector<ScanImage> > camData = sdm->getCameraData();

    bool cam_data_available = camData.size() > 0;

    for (size_t i = 0; i < scans.size(); i++)
    {
        char buf[128];
        std::sprintf(buf, "%05d", scans[i]->positionNumber);
        LVRScanDataItem *item = new LVRScanDataItem(scans[i], sdm, i, m_renderer, QString("pos_") + buf, parent);

        if(cam_data_available && camData[i].size() > 0)
        {
            QTreeWidgetItem* cameras_item = new QTreeWidgetItem(item, LVRCamerasItemType);
            cameras_item->setText(0, QString("Photos"));
            // insert cam poses
            // QTreeWidgetItem *images = new QTreeWidgetItem(item, QString("cams"));
            for(int j=0; j < camData[i].size(); j++)
            {
                char buf2[128];
                std::sprintf(buf2, "%05d", j);
                // implement this
                LVRCamDataItem *cam_item = new LVRCamDataItem(camData[i][j], sdm, j, m_renderer, QString("photo_") + buf2, cameras_item);

                lastItem = cam_item;
            }
        }

        lastItem = item;
    }

    return lastItem;
}

void LVRMainWindow::parseCommandLine(int argc, char** argv)
{

    QStringList filenames;
    viewer::Options options(argc, argv);
    if(options.printUsage())
    {
        return;
    }

    std::vector<std::string> files;
    files.push_back(options.getInputFileName());
    for(int i = 0; i < files.size(); i++)
    {
        std::cout << "filename " << files[i] << std::endl;
        filenames << files[i].c_str();
    }
    
    if(options.isChunkedMesh())
    {
        loadChunkedMesh(filenames, options.getLayers(), options.getCacheSize(),
                        options.getHighResDistance());
    }
    else{
        loadModels(filenames);
    }
}

void LVRMainWindow::changePicker(bool labeling)
{

    if(labeling)
    {
        vtkSmartPointer<vtkAreaPicker> AreaPicker = vtkSmartPointer<vtkAreaPicker>::New();
        qvtkWidget->GetRenderWindow()->GetInteractor()->SetPicker(AreaPicker);
    } else
    {
        vtkSmartPointer<vtkPointPicker> pointPicker = vtkSmartPointer<vtkPointPicker>::New();
        qvtkWidget->GetRenderWindow()->GetInteractor()->SetPicker(pointPicker);
    }
}

void LVRMainWindow::manualLabeling()
{
    if(!m_labeling)
    {
        m_pickingInteractor->labelingOn();

    }else
    {
        m_pickingInteractor->labelingOff();
    }
    m_labeling = !m_labeling;
}
void LVRMainWindow::manualICP()
{
    m_correspondanceDialog->fillComboBoxes();
    m_correspondanceDialog->m_dialog->show();
    m_correspondanceDialog->m_dialog->raise();
    m_correspondanceDialog->m_dialog->activateWindow();
    Q_EMIT(correspondenceDialogOpened());
}

void LVRMainWindow::showColorDialog()
{
    QColor c = QColorDialog::getColor();
    if (c.isValid())
    {
        for (QTreeWidgetItem* item : treeWidget->selectedItems())
        {
            if(item->type() == LVRPointCloudItemType)
            {
                LVRPointCloudItem* pc_item = static_cast<LVRPointCloudItem*>(item);
                pc_item->setColor(c);
            }
            else if(item->type() == LVRMeshItemType)
            {
                LVRMeshItem* mesh_item = static_cast<LVRMeshItem*>(item);
                mesh_item->setColor(c);
            }
            else {
                return;
            }

            highlightBoundingBoxes();
            refreshView();
        }
    }
}

void LVRMainWindow::showTransformationDialog()
{
    buildIncompatibilityBox(string("transformation"), POINTCLOUDS_AND_MESHES_AND_PARENT_ONLY);
    // Get selected item from tree and check type
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();
    if(items.size() > 0)
    {
        QTreeWidgetItem* item = items.first();
        if(item->type() == LVRModelItemType)
        {
            LVRModelItem* item = static_cast<LVRModelItem*>(items.first());
            LVRTransformationDialog* dialog = new LVRTransformationDialog(item, qvtkWidget->GetRenderWindow());
        }
        else if(item->type() == LVRPointCloudItemType || item->type() == LVRMeshItemType)
        {
            if(item->parent()->type() == LVRModelItemType)
            {
                LVRModelItem* l_item = static_cast<LVRModelItem*>(item->parent());
                LVRTransformationDialog* dialog = new LVRTransformationDialog(l_item, qvtkWidget->GetRenderWindow());
            }
            else
            {
                m_incompatibilityBox->exec();
            }
        }
        else
        {
            m_incompatibilityBox->exec();
        }
    }
}

void LVRMainWindow::estimateNormals()
{
    buildIncompatibilityBox(string("normal estimation"), POINTCLOUDS_AND_PARENT_ONLY);
    // Get selected item from tree and check type
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();

    if(items.size() > 0)
    {

        QList<LVRPointCloudItem*> pc_items = getPointCloudItems(items);
        QList<QTreeWidgetItem*> parent_items;
        for(LVRPointCloudItem* pc_item : pc_items)
        {
            parent_items.append(pc_item->parent());
        }

        if(pc_items.size() > 0)
        {
            LVREstimateNormalsDialog* dialog = new LVREstimateNormalsDialog(pc_items, parent_items, treeWidget, qvtkWidget->GetRenderWindow());
            return;
        }
    }
    m_incompatibilityBox->exec();
    qvtkWidget->GetRenderWindow()->Render();
}

void LVRMainWindow::reconstructUsingMarchingCubes()
{
    buildIncompatibilityBox(string("reconstruction"), POINTCLOUDS_AND_PARENT_ONLY);
    // Get selected item from tree and check type
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();
    if(items.size() > 0)
    {
        LVRPointCloudItem* pc_item = getPointCloudItem(items.first());
        QTreeWidgetItem* parent_item = pc_item->parent();
        if(pc_item != NULL)
        {
            LVRReconstructViaMarchingCubesDialog* dialog = new LVRReconstructViaMarchingCubesDialog("MC", pc_item, parent_item, treeWidget, qvtkWidget->GetRenderWindow());
            return;
        }
    }
    m_incompatibilityBox->exec();
}

void LVRMainWindow::reconstructUsingPlanarMarchingCubes()
{
    buildIncompatibilityBox(string("reconstruction"), POINTCLOUDS_AND_PARENT_ONLY);
    // Get selected item from tree and check type
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();
    if(items.size() > 0)
    {
        LVRPointCloudItem* pc_item = getPointCloudItem(items.first());
        LVRModelItem* parent_item = getModelItem(items.first());
        if(pc_item != NULL)
        {
            LVRReconstructViaMarchingCubesDialog* dialog = new LVRReconstructViaMarchingCubesDialog("PMC", pc_item, parent_item, treeWidget, qvtkWidget->GetRenderWindow());
            return;
        }
    }
    m_incompatibilityBox->exec();
}

void LVRMainWindow::reconstructUsingExtendedMarchingCubes()
{
    buildIncompatibilityBox(string("reconstruction"), POINTCLOUDS_AND_PARENT_ONLY);
    // Get selected item from tree and check type
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();
    if(items.size() > 0)
    {
        LVRPointCloudItem* pc_item = getPointCloudItem(items.first());
        LVRModelItem* parent_item = getModelItem(items.first());
        if(pc_item != NULL)
        {
            LVRReconstructViaExtendedMarchingCubesDialog* dialog = new LVRReconstructViaExtendedMarchingCubesDialog("SF", pc_item, parent_item, treeWidget, qvtkWidget->GetRenderWindow());
            return;
        }
    }
    m_incompatibilityBox->exec();
}

void LVRMainWindow::optimizePlanes()
{
    buildIncompatibilityBox(string("planar optimization"), MESHES_AND_PARENT_ONLY);
    // Get selected item from tree and check type
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();
    if(items.size() > 0)
    {
        LVRMeshItem* mesh_item = getMeshItem(items.first());
        LVRModelItem* parent_item = getModelItem(items.first());
        if(mesh_item != NULL)
        {
            LVRPlanarOptimizationDialog* dialog = new LVRPlanarOptimizationDialog(mesh_item, parent_item, treeWidget, qvtkWidget->GetRenderWindow());
            return;
        }
    }
    m_incompatibilityBox->exec();
}

void LVRMainWindow::removeArtifacts()
{
    buildIncompatibilityBox(string("artifact removal"), MESHES_AND_PARENT_ONLY);
    // Get selected item from tree and check type
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();
    if(items.size() > 0)
    {
        LVRMeshItem* mesh_item = getMeshItem(items.first());
        LVRModelItem* parent_item = getModelItem(items.first());
        if(mesh_item != NULL)
        {
            LVRRemoveArtifactsDialog* dialog = new LVRRemoveArtifactsDialog(mesh_item, parent_item, treeWidget, qvtkWidget->GetRenderWindow());
            return;
        }
    }
    m_incompatibilityBox->exec();
}

void LVRMainWindow::applyMLSProjection()
{
    buildIncompatibilityBox(string("MLS projection"), POINTCLOUDS_AND_PARENT_ONLY);
    // Get selected item from tree and check type
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();
    if(items.size() > 0)
    {
        LVRPointCloudItem* pc_item = getPointCloudItem(items.first());
        LVRModelItem* parent_item = getModelItem(items.first());
        if(pc_item != NULL)
        {
            LVRMLSProjectionDialog* dialog = new LVRMLSProjectionDialog(pc_item, parent_item, treeWidget, qvtkWidget->GetRenderWindow());
            return;
        }
    }
    m_incompatibilityBox->exec();
}

void LVRMainWindow::removeOutliers()
{
    buildIncompatibilityBox(string("outlier removal"), POINTCLOUDS_AND_PARENT_ONLY);
    // Get selected item from tree and check type
    QList<QTreeWidgetItem*> items = treeWidget->selectedItems();
    if(items.size() > 0)
    {
        LVRPointCloudItem* pc_item = getPointCloudItem(items.first());
        LVRModelItem* parent_item = getModelItem(items.first());
        if(pc_item != NULL)
        {
            LVRRemoveOutliersDialog* dialog = new LVRRemoveOutliersDialog(pc_item, parent_item, treeWidget, qvtkWidget->GetRenderWindow());
            return;
        }
    }
    m_incompatibilityBox->exec();
}

void LVRMainWindow::buildIncompatibilityBox(string actionName, unsigned char allowedTypes)
{
    // Setup a message box for unsupported items
    string titleString = str(boost::format("Unsupported Item for %1%.") % actionName);
    QString title = QString::fromStdString(titleString);
    string bodyString = "Only %2% are applicable to %1%.";
    QString body;

    if(allowedTypes == MODELITEMS_ONLY)
        bodyString = str(boost::format(bodyString) % actionName % "whole models");
    else if(allowedTypes == POINTCLOUDS_ONLY)
        bodyString = str(boost::format(bodyString) % actionName % "point clouds");
    else if(allowedTypes == MESHES_ONLY)
        bodyString = str(boost::format(bodyString) % actionName % "meshes");
    else if(allowedTypes == POINTCLOUDS_AND_PARENT_ONLY)
        bodyString = str(boost::format(bodyString) % actionName % "point clouds and model items containing point clouds");
    else if(allowedTypes == MESHES_AND_PARENT_ONLY)
        bodyString = str(boost::format(bodyString) % actionName % "meshes and model items containing meshes");
    else if(allowedTypes == POINTCLOUDS_AND_MESHES_AND_PARENT_ONLY)
        bodyString = str(boost::format(bodyString) % actionName % "point clouds, meshes and whole models");

    body = QString::fromStdString(bodyString);

    m_incompatibilityBox->setText(title);
    m_incompatibilityBox->setInformativeText(body);
    m_incompatibilityBox->setStandardButtons(QMessageBox::Ok);
}

void LVRMainWindow::showErrorDialog()
{
    m_errorDialog->show();
    m_errorDialog->raise();
}

void LVRMainWindow::showHistogramDialog()
{
    std::set<LVRPointCloudItem*> pointCloudItems = getSelectedPointCloudItems();
    if(pointCloudItems.empty())
    {
        showErrorDialog();
        return;
    }

    for (LVRPointCloudItem* item : pointCloudItems)
    {
        PointBufferPtr points = item->getPointBuffer();
        if (!points->getUCharChannel("spectral_channels"))
        {
            showErrorDialog();
            return;
        }

        if (!m_histograms.count(item))
        {
            m_histograms[item] = new LVRHistogram(this, points);
        }
        m_histograms[item]->show();
    }
}

void LVRMainWindow::showPointPreview(vtkActor* actor, int point)
{
    if (actor == nullptr || point < 0)
    {
        return;
    }
    LVRPointBufferBridge* pointBridge = nullptr;

    QTreeWidgetItemIterator it(treeWidget);

    while(*it)
    {
        if ((*it)->type() == LVRPointCloudItemType)
        {
            PointBufferBridgePtr pbuf = static_cast<LVRPointCloudItem *>(*it)->getPointBufferBridge();
            if (pbuf->getPointCloudActor() == actor)
            {
                pointBridge = pbuf.get();
                break;
            }
        }
        it++;
    }

    if (pointBridge == nullptr)
    {
        return;
    }
    m_previewPoint = point;
    m_previewPointBuffer = pointBridge->getPointBuffer();
    updatePointPreview(point, pointBridge->getPointBuffer());
}

void LVRMainWindow::showPointInfoDialog()
{
    if (!m_previewPointBuffer)
    {
        return;
    }
    new LVRPointInfo(this, m_previewPointBuffer, m_previewPoint);
}

void LVRMainWindow::onSpectralSliderChanged(int action)
{
    switch(action)
    {
        case QAbstractSlider::SliderSingleStepAdd:
        case QAbstractSlider::SliderSingleStepSub:
        case QAbstractSlider::SliderPageStepAdd:
        case QAbstractSlider::SliderPageStepSub:
        {
            changeSpectralColor();
            break;
        }
        case -1: //valueChanged(int)
        {
            for (int i = 0; i < 3; i++)
            {
                int wavelength = m_spectralSliders[i]->value();
                if (!m_spectralLineEdits[i]->hasFocus())
                {
                    m_spectralLineEdits[i]->setText(QString("%1").arg(wavelength));
                }
            }
        }
    }
}

void LVRMainWindow::onSpectralLineEditSubmit()
{
    onSpectralLineEditChanged();
    changeSpectralColor();
}

void LVRMainWindow::onGradientLineEditSubmit()
{
    onGradientLineEditChanged();
    changeGradientColor();
}

void LVRMainWindow::onGradientLineEditChanged()
{
    std::set<LVRPointCloudItem*> items = getSelectedPointCloudItems();
    if(!items.empty())
    {
        PointBufferPtr points = (*items.begin())->getPointBuffer();
        int min = *points->getIntAtomic("spectral_wavelength_min");
        int max = *points->getIntAtomic("spectral_wavelength_max");


        QString test = m_gradientLineEdit-> text();
        bool ok;
        int wavelength = test.toUInt(&ok);

        if (!ok)
        {
            return;
        }

        if (wavelength < min)
            m_gradientSlider->setValue(min);
        else if (wavelength >= max)
            m_gradientSlider->setValue(max-1);
        else
            m_gradientSlider->setValue(wavelength);

    }
}

void LVRMainWindow::changeSpectralColor()
{
    if (!this->dockWidgetSpectralSliderSettingsContents->isEnabled())
    {
        return;
    }

    std::set<LVRPointCloudItem*> items = getSelectedPointCloudItems();

    if (items.empty())
    {
        return;
    }

    color<size_t> channels;
    color<bool> use_channel;

    PointBufferPtr p = (*items.begin())->getPointBuffer();

    for (int i = 0; i < 3; i++)
    {
        int wavelength = m_spectralSliders[i]->value();
        m_spectralLineEdits[i]->setText(QString("%1").arg(wavelength));

        channels[i] = Util::getSpectralChannel(wavelength, p);

        use_channel[i] = m_spectralCheckboxes[i]->isChecked();
        m_spectralSliders[i]->setEnabled(use_channel[i]);
        m_spectralLineEdits[i]->setEnabled(use_channel[i]);
    }

    for(LVRPointCloudItem* item : items)
    {
        item->getPointBufferBridge()->setSpectralChannels(channels, use_channel);
    }
    m_renderer->GetRenderWindow()->Render();
}

void LVRMainWindow::onSpectralLineEditChanged()
{
    std::set<LVRPointCloudItem*> items = getSelectedPointCloudItems();
    if(!items.empty())
    {
        PointBufferPtr points = (*items.begin())->getPointBuffer();
        int min = *points->getIntAtomic("spectral_wavelength_min");
        int max = *points->getIntAtomic("spectral_wavelength_max");

        for (int i = 0; i < 3; i++)
        {
            QString test = m_spectralLineEdits[i]-> text();
            bool ok;
            int wavelength = test.toUInt(&ok);

            if (!ok)
            {
                return;
            }
            if (wavelength < min)
                m_spectralSliders[i]->setValue(min);
            else if (wavelength >= max)
                m_spectralSliders[i]->setValue(max);
            else
                m_spectralSliders[i]->setValue(wavelength);
        }
    }
}

void LVRMainWindow::onGradientSliderChanged(int action)
{
    switch(action)
    {
        case QAbstractSlider::SliderSingleStepAdd:
        case QAbstractSlider::SliderSingleStepSub:
        case QAbstractSlider::SliderPageStepAdd:
        case QAbstractSlider::SliderPageStepSub:
        {
            changeGradientColor();
            break;
        }
        case -1: //valueChanged(int)
        {
            int wavelength = m_gradientSlider->value();
            if (!m_gradientLineEdit->hasFocus())
            {
                m_gradientLineEdit->setText(QString("%1").arg(wavelength));
            }
        }
    }
}

void LVRMainWindow::changeGradientColor()
{
    if (!this->dockWidgetSpectralColorGradientSettingsContents->isEnabled())
    {
        return;
    }

    std::set<LVRPointCloudItem*> items = getSelectedPointCloudItems();

    if (items.empty())
    {
        return;
    }

    size_t wavelength = m_gradientSlider->value();

    PointBufferPtr p = (*items.begin())->getPointBuffer();

    // @TODO returnvalue could be negative
    size_t channel = Util::getSpectralChannel(wavelength, p);

    bool useNDVI = this->checkBox_NDVI->isChecked();
    bool normalized = this->checkBox_normcolors->isChecked();
    int type = this->comboBox_colorgradient->currentIndex();

    for(LVRPointCloudItem* item : items)
    {
        item->getPointBufferBridge()->setSpectralColorGradient((GradientType)type, channel, normalized, useNDVI);
    }
    m_gradientLineEdit->setText(QString("%1").arg(wavelength));
    m_gradientSlider->setEnabled(!useNDVI);
    m_gradientLineEdit->setEnabled(!useNDVI);
    m_renderer->GetRenderWindow()->Render();
}

void LVRMainWindow::updatePointPreview(int pointId, PointBufferPtr points)
{
    size_t n = points->numPoints();
    points->getPointArray();
    if (pointId < 0 || pointId >= n)
    {
        return;
    }

    size_t n_spec, n_channels;
    UCharChannelOptional spectral_channels = points->getUCharChannel("spectral_channels");

    if (spectral_channels)
    {
        size_t n_spec = spectral_channels->numElements();
        unsigned n_channels = spectral_channels->width();

        if (pointId >= n_spec)
        {
            m_PointPreviewPlotter->removePoints();
        }
        else
        {
            floatArr data(new float[n_channels]);
            for (int i = 0; i < n_channels; i++)
            {
                data[i] = (*spectral_channels)[pointId][i] / 255.0;
            }
            m_PointPreviewPlotter->setPoints(data, n_channels, 0, 1);
            m_PointPreviewPlotter->setXRange(*points->getIntAtomic("spectral_wavelength_min"), *points->getIntAtomic("spectral_wavelength_max"));
        }
    }
}

void LVRMainWindow::updateSpectralSlidersEnabled(bool checked)
{
    if (checked == this->frameSpectralSlidersArea->isEnabled())
    {
        return;
    }

    for (LVRPointCloudItem* item : getSelectedPointCloudItems())
    {
        item->getPointBufferBridge()->useGradient(!checked);
    }
    m_renderer->GetRenderWindow()->Render();

    this->frameSpectralSlidersArea->setEnabled(checked);
    this->frameSpectralGradientArea->setEnabled(!checked);
    this->radioButtonUseSpectralGradient->setChecked(!checked);
}

void LVRMainWindow::updateSpectralGradientEnabled(bool checked)
{
    if (checked == this->frameSpectralGradientArea->isEnabled())
    {
        return;
    }

    for (LVRPointCloudItem* item : getSelectedPointCloudItems())
    {
        item->getPointBufferBridge()->useGradient(checked);
    }
    m_renderer->GetRenderWindow()->Render();

    this->frameSpectralGradientArea->setEnabled(checked);
    this->frameSpectralSlidersArea->setEnabled(!checked);
    this->radioButtonUseSpectralSlider->setChecked(!checked);
}

void LVRMainWindow::updateDisplayLists(actorMap lowRes, actorMap highRes)
{
//    std::unique_lock<std::mutex> lock(m_chunkBridge->mw_mutex);
//    std::cout << "Adding to renderer" << std::endl;
//    m_chunkBridge->release = true;
//    m_chunkBridge->mw_cond.notify_all();
//    lock.unlock();

    for(auto& it: lowRes)
    {
            m_renderer->RemoveActor(it.second);
            it.second->ReleaseGraphicsResources(m_renderer->GetRenderWindow());
    }
    
    for(auto& it: highRes)
    { 
          if(it.second)
          {
              m_renderer->AddActor(it.second);
          }
    }
    m_renderer->GetRenderWindow()->Render();
}
void LVRMainWindow::updatePointCount(uint16_t id, int selectedPointCount)
{

    int topItemCount = labelTreeWidget->topLevelItemCount();
    for (int i = 0; i < topItemCount; i++)
    {
        QTreeWidgetItem* topLevelItem = labelTreeWidget->topLevelItem(i);
        int childCount = topLevelItem->childCount();
        for (int j = 0; j < childCount; j++)
        {
            if(id == topLevelItem->child(j)->data(LABEL_ID_COLUMN, 0).toInt())
            {       
                int pointCountDifference = selectedPointCount - topLevelItem->child(j)->text(LABELED_POINT_COLUMN).toInt();
                topLevelItem->child(j)->setText(LABELED_POINT_COLUMN, QString::number(selectedPointCount));
                //Add points to toplevel points
                topLevelItem->setText(LABELED_POINT_COLUMN, QString::number(pointCountDifference + topLevelItem->text(LABELED_POINT_COLUMN).toInt()));
                return;
            }
        }
    }
}
void LVRMainWindow::cellSelected(QTreeWidgetItem* item, int column)
{
    if(column == LABEL_NAME_COLUMN)
    {
        //Edit Label name
        bool accepted;
        QString label_name = QInputDialog::getText(this, tr("Select Label Name"),
            tr("Label name:"), QLineEdit::Normal,
            item->text(LABEL_NAME_COLUMN), &accepted);
        if (accepted && !label_name.isEmpty())
        {
            item->setText(LABEL_NAME_COLUMN, label_name);
            if (!item->parent())
            {
                //Toplevel item nothing else to do
                return;
            }
            /*
            int comboBoxPos = m_ui->selectedLabelComboBox->findData(item->data(LABEL_ID_COLUMN, 0).toInt());
            if (comboBoxPos >= 0)
            {
                m_ui->selectedLabelComboBox->setItemText(comboBoxPos, label_name);

            }*/
            return;
        }
    }else if(column == LABEL_ID_COLUMN)
    {
        //Change 
        QColor label_color = QColorDialog::getColor(Qt::red, this, tr("Choose Label Color"));
        if (label_color.isValid())
        {
            item->setData(LABEL_ID_COLUMN, 1, label_color);
            if(item->parent())
            {
                //Update Color In picker
                Q_EMIT(labelAdded(item));
                return;
            }
            else
            {
                //ask if all childs Should be updated
    		QMessageBox colorUpdateDialog;
		colorUpdateDialog.setText("Labelclass default color changed. Shall all instance colors be updated?");
		colorUpdateDialog.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
	        colorUpdateDialog.setDefaultButton(QMessageBox::Yes);
                int returnValue = colorUpdateDialog.exec();
                if (returnValue == QMessageBox::Yes)
                {
                    //update All Childs 
                    for (int i = 0; i < item->childCount(); i++)
                    {
                        item->child(i)->setData(LABEL_ID_COLUMN, 1, label_color);
                        Q_EMIT(labelAdded(item->child(i)));
                    }
                }
	
            }
            }
    }
    else
    {
        if(item->parent())
        {
            m_selectedLabelItem = item;

            Q_EMIT(labelChanged(item->data(LABEL_ID_COLUMN, 0).toInt()));
            
        }
    }
}

void LVRMainWindow::visibilityChanged(QTreeWidgetItem* changedItem, int column)
{
    if(column != LABEL_VISIBLE_COLUMN)
    {
        return;
    }

    //check if Instance or whole label changed
    if (changedItem->parent())
    {
        //parent exists item is an instance
        Q_EMIT(hidePoints(changedItem->data(LABEL_ID_COLUMN,0).toInt(), changedItem->checkState(LABEL_VISIBLE_COLUMN)));
    } else
    {
        //Check if unlabeled item
        for (int i = 0; i < changedItem->childCount(); i++)
        {
            QTreeWidgetItem* childItem = changedItem->child(i);

            //sets child elements checkbox on toplevel box value if valuechanged a singal will be emitted and handeled
            childItem->setCheckState(LABEL_VISIBLE_COLUMN, changedItem->checkState(LABEL_VISIBLE_COLUMN));
        }
    }


}

void LVRMainWindow::addNewInstance(QTreeWidgetItem * selectedTopLevelItem)
{
    
    QString choosenLabel = selectedTopLevelItem->text(LABEL_NAME_COLUMN);

    bool accepted;
    QString instance_name = QInputDialog::getText(this, tr("Choose Name for new Instance"),
    tr("Instance name:"), QLineEdit::Normal,
                    QString(choosenLabel + QString::number(selectedTopLevelItem->childCount() + 1)) , &accepted);
    if (!accepted || instance_name.isEmpty())
    {
            //No valid Input
            return;
    }

    QColor label_color = QColorDialog::getColor(selectedTopLevelItem->data(LABEL_ID_COLUMN, 1).value<QColor>(), this, tr("Choose Label Color for first instance"));
    if (!label_color.isValid())
    {
            //Non Valid Color Return 
            return;
    }

    int id = m_id++;
    QTreeWidgetItem * childItem = new QTreeWidgetItem(LVRLabelInstanceType);
    childItem->setText(LABEL_NAME_COLUMN, instance_name);
    childItem->setText(LABELED_POINT_COLUMN, QString::number(0));
    childItem->setCheckState(LABEL_VISIBLE_COLUMN, Qt::Checked);
    childItem->setData(LABEL_ID_COLUMN, 1, label_color);
    childItem->setData(LABEL_ID_COLUMN, 0, id);
    selectedTopLevelItem->addChild(childItem);


    //Add label to combo box 
    selectedInstanceComboBox->addItem(childItem->text(LABEL_NAME_COLUMN), id);
    Q_EMIT(labelAdded(childItem));

}

void LVRMainWindow::loadLabels()
{

    //TODO: What should be done if elements exists?
    QString fileName = QFileDialog::getOpenFileName(this,
                tr("Open HDF5 File"), QDir::homePath(), tr("HDF5 files (*.h5)"));
    if(!QFile::exists(fileName))
    {
        return;
    }

    HDF5Kernel kernel(fileName.toStdString());
    std::vector<std::string> pointCloudNames;
    kernel.subGroupNames("pointclouds", pointCloudNames);
    for (auto pointcloudName : pointCloudNames)
    {
        //pointclouds
        boost::filesystem::path classGroup = (boost::filesystem::path("pointclouds") / boost::filesystem::path(pointcloudName) / boost::filesystem::path("labels"));
        std::vector<std::string> labelClasses;
        kernel.subGroupNames(classGroup.string(), labelClasses);
        for (auto labelClass : labelClasses)
        {
            //Get TopLevel Item for tree view
            QTreeWidgetItem * item = new QTreeWidgetItem();
            item->setText(0, QString::fromStdString(labelClass));
            item->setText(LABELED_POINT_COLUMN, QString::number(0));
            item->setCheckState(LABEL_VISIBLE_COLUMN, Qt::Checked);

            labelTreeWidget->addTopLevelItem(item);   


            //pointclouds/$name/labels/$labelname
            boost::filesystem::path instanceGroup = (classGroup / boost::filesystem::path(labelClass));
            std::vector<std::string> labelInstances;
            kernel.subGroupNames(instanceGroup.string(), labelInstances);
            for (auto instance : labelInstances)
            {

                int id = 0;
                boost::filesystem::path finalGroup = instanceGroup;
                //pointclouds/$name/labels/$labelname/instance
                finalGroup = (instanceGroup / boost::filesystem::path(instance));
                if (labelClass != UNKNOWNNAME)
                {
                    id = m_id++;

                } 

                //Get Color and IDs
                boost::shared_array<int> rgbData;
                std::vector<size_t> rgbDim;
                boost::shared_array<int> idData;
                std::vector<size_t> idDim;
                idData = kernel.loadArray<int>(finalGroup.string(), "IDs", idDim);
                rgbData = kernel.loadArray<int>(finalGroup.string(), "Color", rgbDim);

                //Add Child to top Level
                QTreeWidgetItem * childItem = new QTreeWidgetItem();
                childItem->setText(LABELED_POINT_COLUMN, QString::number(0));
                childItem->setText(0, QString::fromStdString(instance));
                QColor label_color(rgbData[0], rgbData[1], rgbData[2]);
                childItem->setData(LABEL_ID_COLUMN, 1, label_color);
                childItem->setData(LABEL_ID_COLUMN, 0, id);
                item->addChild(childItem);
                Q_EMIT(labelAdded(childItem));
                std::vector<int> out(idData.get(), idData.get() + idDim[0]);
                Q_EMIT(labelLoaded(id, out));
                selectedInstanceComboBox->addItem(childItem->text(LABEL_NAME_COLUMN), id);

            }
        }
    }
}
void LVRMainWindow::exportLabels()
{
    std::vector<uint16_t> labeledPoints = m_pickingInteractor->getLabeles();
    vtkSmartPointer<vtkPolyData> points;
    std::map<uint16_t,std::vector<int>> idMap;

    for (int i = 0; i < labeledPoints.size(); i++)
    {
        
        if(idMap.find(labeledPoints[i]) == idMap.end())
        {
            //first occurence of id add new entry
            idMap[labeledPoints[i]] = {};
        }
        idMap[labeledPoints[i]].push_back(i);
    }
    
    QFileDialog dialog;
    dialog.setDirectory(QDir::homePath());
    dialog.setFileMode(QFileDialog::AnyFile);
    QString strFile = dialog.getSaveFileName(this, "Creat New HDF5 File","","");

    HDF5Kernel label_hdf5kernel((strFile + QString(".h5")).toStdString());
    int topItemCount = labelTreeWidget->topLevelItemCount();


    //TODO This should be for all Pointclouds
    boost::filesystem::path pointcloudName;
    QTreeWidgetItemIterator itu(treeWidget);
    LVRPointCloudItem* citem;
    while (*itu)
    {
        QTreeWidgetItem* item = *itu;

        if ( item->type() == LVRPointCloudItemType)
        {
            citem = static_cast<LVRPointCloudItem*>(*itu);
            pointcloudName = item->parent()->text(0).toStdString();
            points = citem->getPointBufferBridge()->getPolyData();
        }
        itu++;
    }

    double* pointsData = new double[points->GetNumberOfPoints() * 3];
    
    for (int i = 0; i < points->GetNumberOfPoints(); i++)
    {
	auto point = points->GetPoint(i);
        pointsData[(3 * i)] = point[0];
        pointsData[(3 * i) + 1] = point[1];
        pointsData[(3 * i) + 2] = point[2];

    }

    std::vector<size_t> pointsDimension = {3, points->GetNumberOfPoints()};
    boost::shared_array<double> sharedPoints(pointsData);

    //Unlabeled top item
    QTreeWidgetItem* unlabeledItem;
    
    boost::filesystem::path pointGroup = (boost::filesystem::path("pointclouds") / pointcloudName);
    label_hdf5kernel.saveDoubleArray(pointGroup.string(), "Points" , pointsDimension, sharedPoints);
    for (int i = 0; i < topItemCount; i++)
    {
        QTreeWidgetItem* topLevelItem = labelTreeWidget->topLevelItem(i);
        if(topLevelItem->text(LABEL_NAME_COLUMN) == QString::fromStdString(UNKNOWNNAME))
        {
            unlabeledItem = topLevelItem;
        }
        boost::filesystem::path topLabel = topLevelItem->text(LABEL_NAME_COLUMN).toStdString();
        int childCount = topLevelItem->childCount();
        for (int j = 0; j < childCount; j++)
        {
            int childID = topLevelItem->child(j)->data(LABEL_ID_COLUMN, 0).toInt();
            int* sharedArrayData = new int[idMap[childID].size()];
            std::memcpy(sharedArrayData, idMap[childID].data(), idMap[childID].size() * sizeof(int));
            boost::shared_array<int> data(sharedArrayData);
            std::vector<size_t> dimension = {idMap[childID].size()};
            if(idMap.find(childID) != idMap.end())
            { 
                boost::filesystem::path childLabel = (topLevelItem->child(j)->text(LABEL_NAME_COLUMN)).toStdString();
                boost::filesystem::path completeGroup = (pointGroup / boost::filesystem::path("labels") / topLabel / childLabel);

                label_hdf5kernel.saveArray(completeGroup.string(), "IDs" , dimension, data);
                int* rgbSharedData = new int[3];
                (topLevelItem->child(j)->data(LABEL_ID_COLUMN, 1)).value<QColor>().getRgb(&rgbSharedData[0], &rgbSharedData[1], &rgbSharedData[2]);
                boost::shared_array<int> rgbData(rgbSharedData);
                std::vector<size_t> rgbDimension = {3};
                label_hdf5kernel.saveArray(completeGroup.string(), "Color" , rgbDimension, rgbData);
            }
        }
    }
}

void LVRMainWindow::comboBoxIndexChanged(int index)
{
	Q_EMIT(labelChanged(selectedInstanceComboBox->itemData(index).toInt()));
}
} /* namespace lvr2 */
