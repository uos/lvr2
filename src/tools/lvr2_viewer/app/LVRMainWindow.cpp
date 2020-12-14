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
#include <numeric>

#include "LVRMainWindow.hpp"

#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/DataStruct.hpp"
#include "lvr2/io/IOUtils.hpp"
#include "lvr2/io/descriptions/HDF5Kernel.hpp"
#include "lvr2/io/descriptions/LabelHDF5IO.hpp"
#include "lvr2/io/descriptions/HDF5IO.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaHyperlib.hpp"
#include "lvr2/io/descriptions/LabelScanProjectSchemaHDF5V2.hpp"
#include "lvr2/io/descriptions/ScanProjectSchemaHDF5V2.hpp"
#include "lvr2/io/descriptions/DirectoryIO.hpp"

#include "lvr2/io/Polygon.hpp"

#include "lvr2/registration/TransformUtils.hpp"
#include "lvr2/registration/ICPPointAlign.hpp"
#include "lvr2/util/Util.hpp"

#include "../widgets/LVRLabelInstanceTreeItem.hpp"

#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkPointPicker.h>
#include <vtkPointData.h>
#include <vtkAreaPicker.h>
#include <vtkCamera.h>
#include <vtkDefaultPass.h>
#include <vtkCubeSource.h>
#include <vtkAppendPolyData.h>

#include "../vtkBridge/LVRChunkedMeshBridge.hpp"
#include "../vtkBridge/LVRChunkedMeshCuller.hpp"
#include "../vtkBridge/LVRSoilAssistBridge.hpp"

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
    //m_labelDialog = new LVRLabelDialog(treeWidget);
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
    m_actionDeleteLabelClass = new QAction("Delete LabelClass", this);
    m_actionAddNewInstance = new QAction("Add new instance", this);
    m_actionRemoveInstance = new QAction("Remove instance", this);
    m_actionShowWaveform = new QAction("Show Waveform", this);

    m_actionShowImage = new QAction("Show Image", this);
    m_actionSetViewToCamera = new QAction("Set view to camera", this);

    this->addAction(m_actionCopyModelItem);
    this->addAction(m_actionPasteModelItem);

    m_labelTreeParentItemContextMenu = new QMenu();
    m_labelTreeParentItemContextMenu->addAction(m_actionAddLabelClass);
    m_labelTreeParentItemContextMenu->addAction(m_actionAddNewInstance);
    m_labelTreeParentItemContextMenu->addAction(m_actionDeleteLabelClass);
    m_labelTreeChildItemContextMenu = new QMenu();
    m_labelTreeChildItemContextMenu->addAction(m_actionRemoveInstance);
    m_labelTreeChildItemContextMenu->addAction(m_actionAddNewInstance);
    m_labelTreeChildItemContextMenu->addAction(m_actionShowWaveform);

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

#ifdef LVR2_USE_VTK9
    qvtkWidget->renderWindow()->GetInteractor()->SetPicker(pointPicker);
#else
    qvtkWidget->GetRenderWindow()->GetInteractor()->SetPicker(pointPicker);
#endif



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
    /*
    if(m_labelDialog)
    {
        delete m_labelDialog;
    }*/

    if (m_pickingInteractor)
    {
#ifdef LVR2_USE_VTK9
        qvtkWidget->renderWindow()->GetInteractor()->SetInteractorStyle(nullptr);
#else
        qvtkWidget->GetRenderWindow()->GetInteractor()->SetInteractorStyle(nullptr);
#endif

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
    QObject::connect(this->actionOpenScanProject, SIGNAL(triggered()), this, SLOT(openScanProject()));
    QObject::connect(this->actionExportLabeledPointcloud, SIGNAL(triggered()), this, SLOT(exportLabels()));
    QObject::connect(this->actionReadWaveform, SIGNAL(triggered()), this, SLOT(readLWF()));
    QObject::connect(this->actionOpen_SoilAssist, SIGNAL(triggered()), this, SLOT(openSoilAssist()));

    QObject::connect(this->actionExportScanProject, SIGNAL(triggered()), this, SLOT(exportScanProject()));
    QObject::connect(this->actionOpen_Intermedia_Project, SIGNAL(triggered()), this, SLOT(openIntermediaProject()));
    QObject::connect(treeWidget, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(showTreeContextMenu(const QPoint&)));
    QObject::connect(treeWidget, SIGNAL(itemSelectionChanged()), this, SLOT(restoreSliders()));
    QObject::connect(treeWidget, SIGNAL(itemSelectionChanged()), this, SLOT(highlightBoundingBoxes()));
    QObject::connect(treeWidget, SIGNAL(itemChanged(QTreeWidgetItem*, int)), this, SLOT(setModelVisibility(QTreeWidgetItem*, int)));

    QObject::connect(labelTreeWidget, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(showLabelTreeContextMenu(const QPoint&)));


    QObject::connect(m_actionQuit, SIGNAL(triggered()), qApp, SLOT(quit()));
    QObject::connect(this->actionShow_LabelDock, SIGNAL(toggled(bool)), this, SLOT(toggleLabelDock(bool)));

    QObject::connect(m_actionShowColorDialog, SIGNAL(triggered()), this, SLOT(showColorDialog()));
    QObject::connect(m_actionRenameModelItem, SIGNAL(triggered()), this, SLOT(renameModelItem()));
    QObject::connect(m_actionDeleteModelItem, SIGNAL(triggered()), this, SLOT(deleteModelItem()));
    QObject::connect(m_actionCopyModelItem, SIGNAL(triggered()), this, SLOT(copyModelItem()));
    QObject::connect(m_actionPasteModelItem, SIGNAL(triggered()), this, SLOT(pasteModelItem()));
    QObject::connect(m_actionLoadPointCloudData, SIGNAL(triggered()), this, SLOT(loadPointCloudData()));
    QObject::connect(m_actionUnloadPointCloudData, SIGNAL(triggered()), this, SLOT(unloadPointCloudData()));

    QObject::connect(m_actionAddLabelClass, SIGNAL(triggered()), this, SLOT(addLabelClass()));
    QObject::connect(this->addLabelClassButton, SIGNAL(pressed()), this, SLOT(addLabelClass()));

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
    //QObject::connect(m_pickingInteractor, SIGNAL(pointsLabeled(uint16_t, int)), m_labelDialog, SLOT(updatePointCount(uint16_t, int)));
    QObject::connect(m_pickingInteractor, SIGNAL(pointsLabeled(const uint16_t, const int)), this, SLOT(updatePointCount(const uint16_t, const int)));
    QObject::connect(m_pickingInteractor, SIGNAL(lassoSelected()), this->actionSelected_Lasso, SLOT(toggle()));
    QObject::connect(m_pickingInteractor, SIGNAL(polygonSelected()), this->actionSelected_Polygon, SLOT(toggle()));
    //QObject::connect(m_pickingInteractor, SIGNAL(responseLabels(std::vector<uint16_t>)), m_labelDialog, SLOT(responseLabels(std::vector<uint16_t>)));

    //QObject::connect(this, SIGNAL(labelAdded(QTreeWidgetItem*)), m_pickingInteractor, SLOT(newLabel(QTreeWidgetItem*)));
    QObject::connect(this, SIGNAL(hidePoints(int, bool)), m_pickingInteractor, SLOT(setLabeledPointVisibility(int, bool)));

    QObject::connect(labelTreeWidget, SIGNAL(itemChanged(QTreeWidgetItem*, int)), this, SLOT(visibilityChanged(QTreeWidgetItem*, int)));
    //QObject::connect(m_labelDialog, SIGNAL(labelChanged(uint16_t)), m_pickingInteractor, SLOT(labelSelected(uint16_t)));
    QObject::connect(this, SIGNAL(labelChanged(uint16_t)), m_pickingInteractor, SLOT(labelSelected(uint16_t)));

    QObject::connect(labelTreeWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(cellSelected(QTreeWidgetItem*, int)));
    //QObject::connect(m_labelDialog->m_ui->lassotoolButton, SIGNAL(toggled(bool)), m_pickingInteractor, SLOT(setLassoTool(bool)));

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


    //QObject::connect(this->actionSelected_Lasso, SIGNAL(triggered()), this, SLOT(manualLabeling()));
    QObject::connect(this->actionSelected_Lasso, SIGNAL(toggled(bool)), this, SLOT(lassoButtonToggled(bool)));
    //QObject::connect(this->actionSelected_Polygon, SIGNAL(triggered()), this, SLOT(manualLabeling()));
    QObject::connect(this->actionSelected_Polygon, SIGNAL(toggled(bool)), this, SLOT(polygonButtonToggled(bool)));
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

void LVRMainWindow::toggleLabelDock(bool checkBoxState)
{
    if(checkBoxState)
    {
        this->dockWidgetLabel->show();
    }else
    {
        this->dockWidgetLabel->hide();
    }

}
void LVRMainWindow::showBackgroundDialog()
{

#ifdef LVR2_USE_VTK9
    LVRBackgroundDialog dialog(qvtkWidget->renderWindow());
#else
    LVRBackgroundDialog dialog(qvtkWidget->GetRenderWindow());
#endif
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
#ifdef LVR2_USE_VTK9
        this->qvtkWidget->renderWindow()->Render();
#else
        this->qvtkWidget->GetRenderWindow()->Render();
#endif

    }
}

void LVRMainWindow::setupQVTK()
{

#ifdef LVR2_USE_VTK8
    qvtkWidget = new QVTKOpenGLWidget();
#elif defined LVR2_USE_VTK9
    qvtkWidget = new QVTKOpenGLNativeWidget();
#else
    qvtkWidget = new QVTKWidget();
#endif
verticalLayout->replaceWidget(qvtkWidgetPlaceholder,qvtkWidget);
// qvtkWidgetPlaceholder = qvtkWidget;

#if (!defined LVR2_USE_VTK8) && (!defined LVR2_USE_VTK9)
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

#ifdef LVR2_USE_VTK9
    vtkSmartPointer<vtkRenderWindow> renderWindow = this->qvtkWidget->renderWindow();
    m_renderWindowInteractor = this->qvtkWidget->interactor();

#else
    vtkSmartPointer<vtkRenderWindow> renderWindow = this->qvtkWidget->GetRenderWindow();
    m_renderWindowInteractor = this->qvtkWidget->GetInteractor();

#endif

    m_renderWindowInteractor->Initialize();


    // Camera that saves a position that can be loaded
    m_camera = vtkSmartPointer<vtkCamera>::New();

    // Custom interactor to handle picking actions
    //m_pickingInteractor = new LVRPickingInteractor();
    //m_labelInteractor = LVRLabelInteractorStyle::New();
    m_pickingInteractor = LVRPickingInteractor::New();
    m_pickingInteractor->setRenderer(m_renderer);

#ifdef LVR2_USE_VTK9
    qvtkWidget->renderWindow()->GetInteractor()->SetInteractorStyle( m_pickingInteractor );
#else
    qvtkWidget->GetRenderWindow()->GetInteractor()->SetInteractorStyle( m_pickingInteractor );
#endif
   // qvtkWidget->GetRenderWindow()->GetInteractor()->SetInteractorStyle( m_labelInteractor );

    //vtkSmartPointer<vtkPointPicker> pointPicker = vtkSmartPointer<vtkPointPicker>::New();
    vtkSmartPointer<vtkAreaPicker> pointPicker = vtkSmartPointer<vtkAreaPicker>::New();
#ifdef LVR2_USE_VTK9
    qvtkWidget->renderWindow()->GetInteractor()->SetPicker(pointPicker);
#else
    qvtkWidget->GetRenderWindow()->GetInteractor()->SetPicker(pointPicker);
#endif

    // Camera and camera interpolator to be used for camera paths
    m_pathCamera = vtkSmartPointer<vtkCameraRepresentation>::New();
    vtkSmartPointer<vtkCameraInterpolator> cameraInterpolator = vtkSmartPointer<vtkCameraInterpolator>::New();
    cameraInterpolator->SetInterpolationTypeToSpline();
    m_pathCamera->SetInterpolator(cameraInterpolator);
    m_pathCamera->SetCamera(m_renderer->GetActiveCamera());


#ifdef LVR2_USE_VTK_GE_7_1
    // Enable EDL per default
#ifdef LVR2_USE_VTK9
    qvtkWidget->renderWindow()->SetMultiSamples(0);
#else
    qvtkWidget->GetRenderWindow()->SetMultiSamples(0);
#endif

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
#ifdef LVR2_USE_VTK9
    this->qvtkWidget->renderWindow()->Render();
#else
    this->qvtkWidget->GetRenderWindow()->Render();
#endif
#endif
}


void LVRMainWindow::updateView()
{
    m_renderer->ResetCamera();
    m_renderer->ResetCameraClippingRange();
#ifdef LVR2_USE_VTK9
    this->qvtkWidget->renderWindow()->Render();
#else
    this->qvtkWidget->GetRenderWindow()->Render();
#endif

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
#ifdef LVR2_USE_VTK9
    this->qvtkWidget->renderWindow()->Render();
#else
    this->qvtkWidget->GetRenderWindow()->Render();
#endif
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
#ifdef LVR2_USE_VTK9
    this->qvtkWidget->renderWindow()->Render();
#else
    this->qvtkWidget->GetRenderWindow()->Render();
#endif
}

void LVRMainWindow::removeArrow(LVRVtkArrow* a)
{
    if(a)
    {
        m_renderer->RemoveActor(a->getArrowActor());
        m_renderer->RemoveActor(a->getStartActor());
        m_renderer->RemoveActor(a->getEndActor());
    }
#ifdef LVR2_USE_VTK9
    this->qvtkWidget->renderWindow()->Render();
#else
    this->qvtkWidget->GetRenderWindow()->Render();
#endif
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

void LVRMainWindow::deleteLabelInstance(QTreeWidgetItem* item)
{
    m_pickingInteractor->removeLabel(item->data(LABEL_ID_COLUMN, 0).toInt());
    QTreeWidgetItem* parentItem = item->parent();
    LVRLabelClassTreeItem* topLevelItem = static_cast<LVRLabelClassTreeItem* >(parentItem);
    //update the Count avoidign the standart "signal" case to avoid race conditions
    topLevelItem->setText(LABELED_POINT_COLUMN, QString::number(topLevelItem->text(LABELED_POINT_COLUMN).toInt() - item->text(LABELED_POINT_COLUMN).toInt()));
    topLevelItem->removeChild(item);
    //remove the ComboBox entry
    int comboBoxPos = selectedInstanceComboBox->findData(item->data(LABEL_ID_COLUMN, 0).toInt());
    if (comboBoxPos >= 0)
    { 
        selectedInstanceComboBox->removeItem(comboBoxPos);
    }
}
void LVRMainWindow::showLabelTreeContextMenu(const QPoint& p)
{
    QList<QTreeWidgetItem*> items = labelTreeWidget->selectedItems();
    QPoint globalPos = labelTreeWidget->mapToGlobal(p);
    if(items.size() > 0)
    {

        QTreeWidgetItem* item = items.first();
        if (item->type() == LVRLabelClassItemType)
        {
            LVRLabelClassTreeItem *classItem = static_cast<LVRLabelClassTreeItem *>(item);
            auto selected = m_labelTreeParentItemContextMenu->exec(globalPos);
            if(selected == m_actionAddNewInstance)
            {
                addNewInstance(classItem);
            } else if(selected == m_actionDeleteLabelClass)
            {
                int count = item->childCount();
                for(int i = 0; i < count; i++)
                {
                    deleteLabelInstance(item->child(0));
                }
                labelTreeWidget->takeTopLevelItem(labelTreeWidget->indexOfTopLevelItem(item));
            }
            return;
        }
        if(item->type() == LVRLabelInstanceItemType)
        {
            auto selected = m_labelTreeChildItemContextMenu->exec(globalPos);
            if(selected == m_actionRemoveInstance)
            {
                m_pickingInteractor->removeLabel(item->data(LABEL_ID_COLUMN, 0).toInt());
                QTreeWidgetItem* parentItem = item->parent();
                LVRLabelClassTreeItem* topLevelItem = static_cast<LVRLabelClassTreeItem* >(parentItem);
                //update the Count avoidign the standart "signal" case to avoid race conditions
                topLevelItem->setText(LABELED_POINT_COLUMN, QString::number(topLevelItem->text(LABELED_POINT_COLUMN).toInt() - item->text(LABELED_POINT_COLUMN).toInt()));
                topLevelItem->removeChild(item);
                //remove the ComboBox entry
                int comboBoxPos = selectedInstanceComboBox->findData(item->data(LABEL_ID_COLUMN, 0).toInt());
                if (comboBoxPos >= 0)
                { 
                    selectedInstanceComboBox->removeItem(comboBoxPos);
                }

            } else if(selected == m_actionAddNewInstance)
            {
            
                QTreeWidgetItem* parentItem = item->parent();
                LVRLabelClassTreeItem *classItem = static_cast<LVRLabelClassTreeItem *>(parentItem);
                addNewInstance(classItem);
            } else if(selected == m_actionShowWaveform)
            {
                LVRLabelInstanceTreeItem *instanceItem = static_cast<LVRLabelInstanceTreeItem *>(item);
                LabelInstancePtr instancePtr = instanceItem->getInstancePtr();

                //getting all IDS this are the Ids from the combined waveform 
                const auto& totalIds = instancePtr->labeledIDs;
                std::vector<long> combinedWaveform;
                std::vector<int> countWaveform;
                for(const auto& id : totalIds)
                {
                    auto tmp = m_waveformOffset.equal_range(id);
                    auto waveform = std::prev(tmp.first, 1)->second;
                    uint32_t pcId = id - std::prev(tmp.first, 1)->first;

                    for (int i = 0; i < waveform->waveformIndices[pcId + 1] - waveform->waveformIndices[pcId]; i++)
                    {
                        if(combinedWaveform.size() <= i)
                        {
                            combinedWaveform.push_back(waveform->waveformSamples[pcId + i]);
                            countWaveform.push_back(1);;
                        }
                        else
                        {
                            combinedWaveform[i] += waveform->waveformSamples[pcId + i];
                            countWaveform[i]++;
                        }
                    }
                }
                /*
                Channel<uint16_t>::Optional opt = labelTreeWidget->getLabelRoot()->points->getChannel<uint16_t>("waveform");
                if (opt)
                {
                    size_t n = opt->numElements();
                    size_t width = opt->width();
                    boost::shared_array<uint16_t> waveforms = opt->dataPtr();

                    //get Selected Ids
                    std::vector<int> labelID;
                    std::vector<uint16_t> labeledPoints = m_pickingInteractor->getLabeles();
                    int index = item->data(LABEL_ID_COLUMN, 0).toInt();

                    for (int i = 0; i < labeledPoints.size(); i++)
                    {
                        if (index == labeledPoints[i])
                        {
                            labelID.push_back(i);
                        }
                    }

                    std::vector<int> combinedWaveform;
                    combinedWaveform.resize(width);
                    for (auto id : labelID)
                    {
                        for (int i = 0; i < width; i++)
                        {
                            combinedWaveform[i] += waveforms[(id * width) + i];
                        }
                    }
                    */
                    floatArr plotData(new float[combinedWaveform.size()]);
                    LVRPlotter* plotter =new LVRPlotter;
                    for(int i = 0; i < combinedWaveform.size(); i++)
                    {
                        plotData[i] = (combinedWaveform[i] / countWaveform[i]);
                    }
                    plotter->setPoints(plotData, combinedWaveform.size());
                    plotter->setXRange(0, combinedWaveform.size());
                    QDialog window(this);
                    QHBoxLayout *HLayout = new QHBoxLayout(&window);
                    HLayout->addWidget(plotter);
                    window.setLayout(HLayout);
                    window.exec();

                /*} else
                {
                    QMessageBox noWaveformDialog;
                    noWaveformDialog.setText("No Waveform found.");
                    noWaveformDialog.setStandardButtons(QMessageBox::Ok);
                    noWaveformDialog.setIcon(QMessageBox::Warning);
                    int returnValue = noWaveformDialog.exec();
                }*/
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
    if(!m_pickingInteractor->getPoints())
    {
        //Set Points for pickinginteractor if needed
        QStringList pointcloudNames;
        std::vector<LVRPointCloudItem*> pointcloudItems;
    	QTreeWidgetItemIterator itu(treeWidget);
        LVRPointCloudItem* citem;
        std::map<QString, LVRPointCloudItem*> pointclouds;
	
        //check if a Scan Project is loaded
        if(!checkForScanProject())
        {
            return;
        }
        

        while (*itu)
        {
            QTreeWidgetItem* item = *itu;

            if ( item->type() == LVRPointCloudItemType)
            {
                citem = static_cast<LVRPointCloudItem*>(*itu);
                QString key = item->parent()->parent()->text(0) + "\\" + item->parent()->text(0);
                pointclouds[key] = citem ;
                pointcloudNames << key; 
            }
            itu++;
        }
        LVRPointcloudSelectionDialog pcDialog(pointcloudNames);
        if (pointcloudNames.size() > 1)
        {
            if(pcDialog.exec() != QDialog::Accepted)
            {
                return;
            }
        }

        size_t pcSize = 0;
        std::vector<WaveformPtr> partWaveforms;
        for(const auto& pcName : pcDialog.getChecked())
        {
            auto parts = pcName.split(QString("\\"));
            std::pair<uint32_t, uint32_t> key;
            std::pair<std::pair<uint32_t, uint32_t>, uint32_t> entry;
            key = std::make_pair(parts[0].toUInt(),parts[1].toUInt());

            citem = pointclouds[pcName];
            if(citem->parent() && citem->parent()->type() == LVRModelItemType)
            {
                LVRModelItem* pitem = static_cast<LVRModelItem*>(citem->parent());
                if(pitem->getModelBridge()->getWaveform())
                {
                    m_waveformOffset[pcSize] = pitem->getModelBridge()->getWaveform();
                    partWaveforms.push_back(pitem->getModelBridge()->getWaveform());
                }

            }
            //entry = std::make_pair(key, pcSize);
            pcSize += citem->getPointBuffer()->numPoints();
            labelTreeWidget->getLabelRoot()->pointOffsets.push_back(entry); 
        }

        floatArr combinedPointcloud = floatArr(new float[pcSize * 3]);
        size_t iterator = 0;
        for(const auto& pcName : pcDialog.getChecked())
        {
            citem = pointclouds[pcName];
            std::memcpy(combinedPointcloud.get() + iterator, citem->getPointBuffer()->getPointArray().get(), citem->getPointBuffer()->numPoints() * 3 * sizeof(float));
            iterator += (citem->getPointBuffer()->numPoints() * 3);
        }

        PointBufferPtr pb = PointBufferPtr(new PointBuffer(combinedPointcloud, pcSize));
        PointBufferBridgePtr temp = PointBufferBridgePtr(new LVRPointBufferBridge(pb));
        m_pickingInteractor->setPoints(temp->getPolyIDData());
        labelTreeWidget->getLabelRoot()->points = temp->getPointBuffer();


        //add Waveform
        if (m_waveformOffset.size() > 0)
        {
            WaveformPtr waveform = WaveformPtr(new Waveform);
            std::vector<long> indices= {0};
            int maxBucket = 0;
            for(const auto partWaveform : partWaveforms)
            {
                maxBucket = std::max(maxBucket,partWaveform->maxBucketSize);
                long offset = indices.back();
                waveform->waveformSamples.insert(waveform->waveformSamples.end(), partWaveform->waveformSamples.begin(), partWaveform->waveformSamples.end());
                waveform->lowPower.insert(waveform->lowPower.end(), partWaveform->lowPower.begin(), partWaveform->lowPower.end());
                waveform->echoType.insert(waveform->echoType.end(), partWaveform->echoType.begin(), partWaveform->echoType.end());
                std::transform(partWaveform->waveformIndices.begin() + 1, partWaveform->waveformIndices.end(), std::back_inserter(indices), [&](int i){return i + offset;});
            }
            waveform->waveformIndices = std::move(indices);
            waveform->maxBucketSize = maxBucket;
            labelTreeWidget->getLabelRoot()->waveform = waveform;
        }
        
    }

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

    QColor labelColor = QColorDialog::getColor(Qt::red, this, tr("Choose default Label Color for label Class(willbe used for first isntance)"));
    if (!labelColor.isValid())
    {
            //Non Valid Color Return 
            return;
    }

    if (labelTreeWidget->topLevelItemCount() == 0)
    {

        //Create Unlabeled Class
        LVRLabelClassTreeItem * classItem = new LVRLabelClassTreeItem(UNKNOWNNAME, 0, true, true, QColor(Qt::red));
        // and instance
        LVRLabelInstanceTreeItem * instanceItem = new LVRLabelInstanceTreeItem(UNKNOWNNAME, 0, 0 , true, true, QColor(Qt::red));

        classItem->addChild(instanceItem);
        labelTreeWidget->addTopLevelItem(classItem);    
        //Q_EMIT(labelAdded(instanceItem));
        m_pickingInteractor->newLabel(instanceItem);
        selectedInstanceComboBox->addItem(QString::fromStdString(instanceItem->getName()), 0);

        std::vector<int> out;
        QTreeWidgetItemIterator itu(treeWidget);
        LVRPointCloudItem* citem;
        while (*itu)
        {
            QTreeWidgetItem* item = *itu;

            if ( item->type() == LVRPointCloudItemType)
            {
                citem = static_cast<LVRPointCloudItem*>(*itu);
                out = std::vector<int>(citem->getPointBufferBridge()->getPolyData()->GetNumberOfPoints());
            }
            itu++;
        }
        std::iota(out.begin(), out.end(), 0);
        m_pickingInteractor->setLabel(0, out);
    }

    int id = labelTreeWidget->getNextId();
    //Setting up new Toplevel item
    LVRLabelClassTreeItem * classItem = new LVRLabelClassTreeItem(className.toStdString(), 0, true, true, labelColor);
    LVRLabelInstanceTreeItem * instanceItem = new LVRLabelInstanceTreeItem((className.toStdString() + "0"), id, 0 , true, true, labelColor);
    classItem->addChild(instanceItem);
    m_selectedLabelItem = instanceItem;
    labelTreeWidget->addTopLevelItem(classItem);    
    
    //Add label to combo box 
    selectedInstanceComboBox->addItem(QString::fromStdString(instanceItem->getName()), id);
   // Q_EMIT(labelAdded(instanceItem));
    m_pickingInteractor->newLabel(instanceItem);
    int comboBoxPos = selectedInstanceComboBox->findData(instanceItem->getId());
    selectedInstanceComboBox->setCurrentIndex(comboBoxPos);
    //Q_EMIT(labelChanged(id));
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
    // Load model and generate vtk representation
	std::cout << "Read Model " << std::endl;
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
	    if(info.suffix() == "")
	    {
                //read intermediaformat
                DirectoryKernelPtr dirKernelPtr(new DirectoryKernel(info.absoluteFilePath().toStdString())); 
                std::string tmp = info.absolutePath().toStdString();
                DirectorySchemaPtr hyperlibSchemaPtr(new ScanProjectSchemaHyperlib(tmp)); 
                DirectoryIO dirIO(dirKernelPtr, hyperlibSchemaPtr);
                ScanProjectPtr scanProject = dirIO.loadScanProject();
                ScanProjectBridgePtr bridge(new LVRScanProjectBridge(scanProject));
                bridge->addActors(m_renderer);
                LVRScanProjectItem* item = new LVRScanProjectItem(bridge, "ScanProject");
                QTreeWidgetItem *root = new QTreeWidgetItem(treeWidget);
                root->addChild(item);
                item->setExpanded(false);
            //lastItem = item;
            }else if (info.suffix() == "h5")
            {
                openHDF5(info.absoluteFilePath().toStdString());
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

			//m_pickingInteractor->setPoints(citem->getPointBufferBridge()->getPolyIDData());
            //labelTreeWidget->getLabelRoot()->points = citem->getPointBuffer();
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

void LVRMainWindow::loadScanProjectDir()
{
    // Commented out due to different type errors

    // QString dirPath = QFileDialog::getExistingDirectory(this, tr("Open Directory"), "", QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    // DirectorySchemaPtr hyperlibSchema(new ScanProjectSchemaHyperlib);
    // //DirectorySchemaPtr slamSchemaPtr(new ScanProjectSchemaSLAM);
    // DirectoryKernelPtr slamDirKernel(new DirectoryKernel(dirPath));
    // DirectoryIO slamIO(slamDirKernel, hyperlibSchema);
    // ScanProjectPtr slamProject = slamIO.loadScanProject();
    
    // std::vector<ScanPositionPtr> positions = slamProject->positions;
    // std::vector<ScanPtr> scans = positions[0]->scans;
    // std::cout << "test" << std::endl;


    // std::cout << positions.size() << std::endl;

}

void LVRMainWindow::loadScanProjectH5()
{
    QMessageBox msgBox;
    msgBox.setText("Hello World H5");
    msgBox.exec();
}

void LVRMainWindow::loadScanProject()
{
    QMessageBox msgBox;
    msgBox.setText("Hello World");
    msgBox.exec();
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
    else if (treeWidgetItem->type() == LVRLabeledScanProjectEditMarkItemType)
    {
        LVRLabeledScanProjectEditMarkItem *item = static_cast<LVRLabeledScanProjectEditMarkItem *>(treeWidgetItem);
        item->setVisibility(m_actionShow_Points->isChecked());

        refreshView();
    }
    else if (treeWidgetItem->type() == LVRScanPositionItemType)
    {
        LVRScanPositionItem *item = static_cast<LVRScanPositionItem *>(treeWidgetItem);
        item->setModelVisibility(0,m_actionShow_Points->isChecked());
        refreshView();
    }
      else if (treeWidgetItem->type() == LVRLabelItemType)
    {
        LVRLabelItem *item = static_cast<LVRLabelItem *>(treeWidgetItem);
        item->setVisibility(m_actionShow_Points->isChecked());
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
            if (item->parent()->type() == LVRLabelItemType)
            {
                LVRLabelItem* label_item = static_cast<LVRLabelItem*>(item->parent());
                if(label_item->isEnabled()) label_item->setVisibility(checkboxState);
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
#ifdef LVR2_USE_VTK9
        qvtkWidget->renderWindow()->GetInteractor()->SetPicker(AreaPicker);
#else
        qvtkWidget->GetRenderWindow()->GetInteractor()->SetPicker(AreaPicker);
#endif
    } else
    {
        vtkSmartPointer<vtkPointPicker> pointPicker = vtkSmartPointer<vtkPointPicker>::New();
#ifdef LVR2_USE_VTK9
        qvtkWidget->renderWindow()->GetInteractor()->SetPicker(pointPicker);
#else
        qvtkWidget->GetRenderWindow()->GetInteractor()->SetPicker(pointPicker);
#endif
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
#ifdef LVR2_USE_VTK9
            LVRTransformationDialog* dialog = new LVRTransformationDialog(item, qvtkWidget->renderWindow());
#else
            LVRTransformationDialog* dialog = new LVRTransformationDialog(item, qvtkWidget->GetRenderWindow());
#endif
        }
        else if(item->type() == LVRPointCloudItemType || item->type() == LVRMeshItemType)
        {
            if(item->parent()->type() == LVRModelItemType)
            {
                LVRModelItem* l_item = static_cast<LVRModelItem*>(item->parent());
#ifdef LVR2_USE_VTK9
                LVRTransformationDialog* dialog = new LVRTransformationDialog(l_item, qvtkWidget->renderWindow());
#else
                LVRTransformationDialog* dialog = new LVRTransformationDialog(l_item, qvtkWidget->GetRenderWindow());
#endif
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
#ifdef LVR2_USE_VTK9
            LVREstimateNormalsDialog* dialog = new LVREstimateNormalsDialog(pc_items, parent_items, treeWidget, qvtkWidget->renderWindow());
#else
            LVREstimateNormalsDialog* dialog = new LVREstimateNormalsDialog(pc_items, parent_items, treeWidget, qvtkWidget->GetRenderWindow());
#endif
            return;
        }
    }
    m_incompatibilityBox->exec();
#ifdef LVR2_USE_VTK9
    qvtkWidget->renderWindow()->Render();
#else
    qvtkWidget->GetRenderWindow()->Render();
#endif
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
#ifdef LVR2_USE_VTK9
            LVRReconstructViaMarchingCubesDialog* dialog = new LVRReconstructViaMarchingCubesDialog("MC", pc_item, parent_item, treeWidget, qvtkWidget->renderWindow());
#else
            LVRReconstructViaMarchingCubesDialog* dialog = new LVRReconstructViaMarchingCubesDialog("MC", pc_item, parent_item, treeWidget, qvtkWidget->GetRenderWindow());
#endif
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
#ifdef LVR2_USE_VTK9
            LVRReconstructViaMarchingCubesDialog* dialog = new LVRReconstructViaMarchingCubesDialog("PMC", pc_item, parent_item, treeWidget, qvtkWidget->renderWindow());
#else
            LVRReconstructViaMarchingCubesDialog* dialog = new LVRReconstructViaMarchingCubesDialog("PMC", pc_item, parent_item, treeWidget, qvtkWidget->GetRenderWindow());
#endif
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
#ifdef LVR2_USE_VTK9
            LVRReconstructViaExtendedMarchingCubesDialog* dialog = new LVRReconstructViaExtendedMarchingCubesDialog("SF", pc_item, parent_item, treeWidget, qvtkWidget->renderWindow());
#else
            LVRReconstructViaExtendedMarchingCubesDialog* dialog = new LVRReconstructViaExtendedMarchingCubesDialog("SF", pc_item, parent_item, treeWidget, qvtkWidget->GetRenderWindow());
#endif
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
#ifdef LVR2_USE_VTK9
            LVRPlanarOptimizationDialog* dialog = new LVRPlanarOptimizationDialog(mesh_item, parent_item, treeWidget, qvtkWidget->renderWindow());
#else
            LVRPlanarOptimizationDialog* dialog = new LVRPlanarOptimizationDialog(mesh_item, parent_item, treeWidget, qvtkWidget->GetRenderWindow());
#endif
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
#ifdef LVR2_USE_VTK9
            LVRRemoveArtifactsDialog* dialog = new LVRRemoveArtifactsDialog(mesh_item, parent_item, treeWidget, qvtkWidget->renderWindow());
#else
            LVRRemoveArtifactsDialog* dialog = new LVRRemoveArtifactsDialog(mesh_item, parent_item, treeWidget, qvtkWidget->GetRenderWindow());
#endif
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
#ifdef LVR2_USE_VTK9
            LVRMLSProjectionDialog* dialog = new LVRMLSProjectionDialog(pc_item, parent_item, treeWidget, qvtkWidget->renderWindow());
#else
            LVRMLSProjectionDialog* dialog = new LVRMLSProjectionDialog(pc_item, parent_item, treeWidget, qvtkWidget->GetRenderWindow());
#endif
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
#ifdef LVR2_USE_VTK9
            LVRRemoveOutliersDialog* dialog = new LVRRemoveOutliersDialog(pc_item, parent_item, treeWidget, qvtkWidget->renderWindow());
#else
            LVRRemoveOutliersDialog* dialog = new LVRRemoveOutliersDialog(pc_item, parent_item, treeWidget, qvtkWidget->GetRenderWindow());
#endif
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

	    //update comboxBoxItem
            int comboBoxPos = selectedInstanceComboBox->findData(item->data(LABEL_ID_COLUMN, 0).toInt());
            if (comboBoxPos >= 0)
            {
                selectedInstanceComboBox->setItemText(comboBoxPos, label_name);
            }
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
                //Q_EMIT(labelAdded(item));
                m_pickingInteractor->newLabel(item);
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
                       // Q_EMIT(labelAdded(item->child(i)));
                        m_pickingInteractor->newLabel(item->child(i));
                    }
                }
	
            }
            }
    }
    else
    {
        if(item->parent())
        {
            //Element double Clicked change selected Label
            m_selectedLabelItem = item;
            int comboCount = selectedInstanceComboBox->count();
            for (int i = 0; i < comboCount; i++)
            {
                int comboBoxid = selectedInstanceComboBox->itemData(i).toInt();
                if(item->data(LABEL_ID_COLUMN,0).toInt() == comboBoxid)
                {
                    selectedInstanceComboBox->setCurrentIndex(i);
                }
            }
            
        }
    }
}

void LVRMainWindow::visibilityChanged(QTreeWidgetItem* changedItem, int column)
{
    if (column == LABEL_VISIBLE_COLUMN || column == LABEL_EDITABLE_COLUMN)
    {
        //check if Instance or whole label changed
	if (changedItem->parent())
	{
	    if(column == LABEL_VISIBLE_COLUMN)
	    {
	    	//parent exists item is an instance
	    	Q_EMIT(hidePoints(changedItem->data(LABEL_ID_COLUMN,0).toInt(), changedItem->checkState(LABEL_VISIBLE_COLUMN)));
	    } 
	    else if (column == LABEL_EDITABLE_COLUMN)
	    {
		//m_pickingInteractor->setEditability(changedItem->data(LABEL_ID_COLUMN,0).toInt(), changedItem->checkState(LABEL_EDITABLE_COLUMN));
	    }
	} else
	{
		//Check if unlabeled item
		for (int i = 0; i < changedItem->childCount(); i++)
	    {
	        QTreeWidgetItem* childItem = changedItem->child(i);

	        //sets child elements checkbox on toplevel box value if valuechanged a singal will be emitted and handeled
	        childItem->setCheckState(column, changedItem->checkState(column));
	    }
        }
    }
}

void LVRMainWindow::addNewInstance(LVRLabelClassTreeItem * selectedTopLevelItem)
{
    
    QString choosenLabel = selectedTopLevelItem->text(LABEL_NAME_COLUMN);

    bool accepted;
    QString instanceName = QInputDialog::getText(this, tr("Choose Name for new Instance"),
    tr("Instance name:"), QLineEdit::Normal,
                    QString(choosenLabel + QString::number(selectedTopLevelItem->childCount() + 1)) , &accepted);
    if (!accepted || instanceName.isEmpty())
    {
            //No valid Input
            return;
    }

    QColor labelColor = QColorDialog::getColor(selectedTopLevelItem->data(LABEL_ID_COLUMN, 1).value<QColor>(), this, tr("Choose Label Color for first instance"));
    if (!labelColor.isValid())
    {
            //Non Valid Color Return 
            return;
    }

    int id = labelTreeWidget->getNextId();
    LVRLabelInstanceTreeItem * instanceItem = new LVRLabelInstanceTreeItem(instanceName.toStdString(), id, 0, true, true, labelColor);
    selectedTopLevelItem->addChild(instanceItem);

    //Add label to combo box 
    selectedInstanceComboBox->addItem(QString::fromStdString(instanceItem->getName()), id);
    m_pickingInteractor->newLabel(instanceItem);
    //Q_EMIT(labelAdded(childItem));

    int comboBoxPos = selectedInstanceComboBox->findData(instanceItem->getId());
    selectedInstanceComboBox->setCurrentIndex(comboBoxPos);
}
void LVRMainWindow::openIntermediaProject()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("Choose IntermediaProject"),"",QFileDialog::ShowDirsOnly| QFileDialog::DontResolveSymlinks);
    if (dir == "")
    {
        //Selection Was Canceled
        return;
    }
    DirectoryKernelPtr dirKernelPtr(new DirectoryKernel(dir.toStdString())); 
    std::string tmp  = dir.toStdString();
    DirectorySchemaPtr hyperlibSchemaPtr(new ScanProjectSchemaHyperlib(tmp)); 
    DirectoryIO dirIO(dirKernelPtr, hyperlibSchemaPtr);
    ScanProjectPtr scanProject = dirIO.loadScanProject();
    LabeledScanProjectEditMarkPtr labelScanProject= LabeledScanProjectEditMarkPtr(new LabeledScanProjectEditMark());
    ScanProjectEditMarkPtr editMark = ScanProjectEditMarkPtr(new ScanProjectEditMark());
    editMark->project = scanProject;
    labelScanProject->editMarkProject = editMark;
    this->treeWidget->addLabeledScanProjectEditMark(labelScanProject, dir.toStdString());
    //this->treeWidget->addScanProject(scanProject, dir.toStdString());
    this->treeWidget->getBridgePtr()->addActors(m_renderer);
    updateView();
    labelScanProject->labelRoot = labelTreeWidget->getLabelRoot();

}
void LVRMainWindow::openScanProject()
{
    // QString fileName = QFileDialog::getOpenFileName(this,
    //             tr("Open HDF5 File"), QDir::homePath(), tr("HDF5 files (*.h5)"));
    // if(!QFile::exists(fileName))
    // {
    //     return;
    // }
    // openHDF5(fileName.toStdString());

    LVRScanProjectOpenDialog dialog(this);
    
    
}
 
void LVRMainWindow::openHDF5(std::string fileName)
{
    LabelHDF5SchemaPtr hdf5Schema(new LabelScanProjectSchemaHDF5V2);
    HDF5KernelPtr hdf5Kernel(new HDF5Kernel(fileName));
    LabelHDF5IO h5IO(hdf5Kernel, hdf5Schema);
    LabeledScanProjectEditMarkPtr labelScanProject = h5IO.loadScanProject();
   
    this->treeWidget->addLabeledScanProjectEditMark(labelScanProject, fileName);

    if(labelScanProject->labelRoot)
    {
        this->dockWidgetLabel->show();
        m_pickingInteractor->setPoints(this->treeWidget->getBridgePtr()->getLabelBridgePtr()->getPointBridge()->getPolyIDData());
        this->labelTreeWidget->setLabelRoot(labelScanProject->labelRoot, m_pickingInteractor,selectedInstanceComboBox);
    } else
    {
        labelScanProject->labelRoot = labelTreeWidget->getLabelRoot();
    }
    this->treeWidget->getBridgePtr()->addActors(m_renderer);
    updateView();


   
}
void LVRMainWindow::exportLabels()
{
    std::cout << "Unused" << std::endl;
    std::vector<uint16_t> labeledPoints = m_pickingInteractor->getLabeles();
    vtkSmartPointer<vtkPolyData> points;
    std::map<uint16_t, std::vector<int>> idMap;

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

    std::vector<size_t> pointsDimension = {3, static_cast<long unsigned int>(points->GetNumberOfPoints())};
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

    LVRModelItem* lvrmodel = getModelItem(treeWidget->topLevelItem(0));
    ModelBridgePtr bridge = lvrmodel->getModelBridge();
    PointBufferBridgePtr pointBridge = bridge->getPointBridge();
    PointBufferPtr pointBuffer = pointBridge->getPointBuffer();

    /*
    //Waveform
    boost::filesystem::path waveGroup = (pointGroup / "Waveformdata");
    Channel<uint16_t>::Optional optInt = pointBuffer->getChannel<uint16_t>("waveform");
    if (optInt)
    {
	    size_t n = optInt->numElements();
	    size_t width = optInt->width();
            std::vector<size_t> dimension = {n, width};
            label_hdf5kernel.saveArray(waveGroup.string(), "Waveform" , dimension, optInt->dataPtr());
    }
    Channel<float>::Optional opt = pointBuffer->getChannel<float>("amplitude");
    if (opt)
    {
	    size_t n = opt->numElements();
	    size_t width = opt->width();
            std::vector<size_t> dimension = {n, width};
            label_hdf5kernel.saveArray(waveGroup.string(), "Amplitude" , dimension, opt->dataPtr());
    }
    opt = pointBuffer->getChannel<float>("deviation");
    if (opt)
    {
	    size_t n = opt->numElements();
	    size_t width = opt->width();
            std::vector<size_t> dimension = {n, width};
            label_hdf5kernel.saveArray(waveGroup.string(), "deviation" , dimension, opt->dataPtr());
    }
    opt = pointBuffer->getChannel<float>("reflectance");
    if (opt)
    {
	    size_t n = opt->numElements();
	    size_t width = opt->width();
            std::vector<size_t> dimension = {n, width};
            label_hdf5kernel.saveArray(waveGroup.string(), "reflectance" , dimension, opt->dataPtr());
    }
    opt = pointBuffer->getChannel<float>("backgroundRadiationChannel");
    if (opt)
    {
	    size_t n = opt->numElements();
	    size_t width = opt->width();
            std::vector<size_t> dimension = {n, width};
            label_hdf5kernel.saveArray(waveGroup.string(), "backgroundRadiation" , dimension, opt->dataPtr());
    }
    */

}

void LVRMainWindow::comboBoxIndexChanged(int index)
{
    labelTreeWidget->itemSelected(selectedInstanceComboBox->itemData(index).toInt());
    Q_EMIT(labelChanged(selectedInstanceComboBox->itemData(index).toInt()));
}

void LVRMainWindow::lassoButtonToggled(bool checked)
{
    if (checked)
    {
        if(labelTreeWidget->topLevelItemCount() == 0)
        {
            //NO label was created - show warning
            const QSignalBlocker blocker(this->actionSelected_Lasso);
            QMessageBox noLabelDialog;
            noLabelDialog.setText("No Label Instance was created! Create an instance beofre labeling Points.");
            noLabelDialog.setStandardButtons(QMessageBox::Ok);
            noLabelDialog.setIcon(QMessageBox::Warning);
            int returnValue = noLabelDialog.exec();
            this->actionSelected_Lasso->setChecked(false);

            return;
        }
	//setLasso Tool
        m_pickingInteractor->setLassoTool(true);
        //check if Polygon tool was enabled
        if (this->actionSelected_Polygon->isChecked())
        {
            const QSignalBlocker blocker(this->actionSelected_Polygon);
            this->actionSelected_Polygon->setChecked(false);
	    return;
        }
        m_pickingInteractor->labelingOn();
    } else
    {
        m_pickingInteractor->labelingOff();
    }
}
void LVRMainWindow::polygonButtonToggled(bool checked)
{

    if (checked)
    {
        if(labelTreeWidget->topLevelItemCount() == 0)
        {
            //NO label was created - show warning
            const QSignalBlocker blocker(this->actionSelected_Polygon);
            QMessageBox noLabelDialog;
            noLabelDialog.setText("No Label Instance was created! Create an instance beofre labeling Points.");
            noLabelDialog.setStandardButtons(QMessageBox::Ok);
            noLabelDialog.setIcon(QMessageBox::Warning);
            int returnValue = noLabelDialog.exec();
            this->actionSelected_Polygon->setChecked(false);
            return;
        }
	//setPolygonTool
        m_pickingInteractor->setLassoTool(false);
        //check if lasso tool was enabled
        if (this->actionSelected_Lasso->isChecked())
        {
            const QSignalBlocker blocker(this->actionSelected_Lasso);
            this->actionSelected_Lasso->setChecked(false);
	    return;
        }
        m_pickingInteractor->labelingOn();
    } else
    {
        m_pickingInteractor->labelingOff();
    }

}

void LVRMainWindow::openSoilAssist()
{
    QString fileName = QFileDialog::getOpenFileName(this,
            tr("Open HDF5 File"), QDir::homePath(), tr("HDF5 files (*.h5)"));
    if(!QFile::exists(fileName))
    {
        return;
    }

    SoilAssistFieldPtr field(new SoilAssistField);
    field->fromH5File(fileName.toStdString());

    SoilAssistBridgePtr polybrdige(new LVRSoilAssistBridge(field));
    for(auto && actor : polybrdige->getPolygonActors())
    {
        m_renderer->AddActor(actor);

    }

    highlightBoundingBoxes();
    restoreSliders();
    assertToggles();
    updateView();
}

void LVRMainWindow::readLWF()
{

    LVRLabeledScanProjectEditMarkItem* projectItem;
    if (!(projectItem = checkForScanProject()))
    {
        return;
    }

    QString fileName = QFileDialog::getOpenFileName(this,
                tr("Select LWF File"), QDir::homePath(), tr("LASVegasWaveformFiles(*.lwf)"));
    std::ifstream waveformFile(fileName.toStdString(), std::ios::in | std::ios::binary);
    if (!waveformFile)
    {
        QMessageBox warning;
        warning.setText("Could not open File");
        warning.setStandardButtons(QMessageBox::Ok);
        warning.setIcon(QMessageBox::Warning);
        warning.exec();
        return;
    }
    uint64_t dimSize, lines, colums;
    waveformFile.read((char *) &dimSize, sizeof(unsigned long long));
    if (dimSize != 2)
    {
        return;
        //TODO show Warning
    }
    waveformFile.read((char *) &lines, sizeof(unsigned long long));
    waveformFile.read((char *) &colums, sizeof(unsigned long long));

    WaveformPtr waveform = WaveformPtr(new Waveform);
    waveform->waveformIndices.push_back(0);
    uint16_t data[colums];

    for (int i = 0; i < lines; i++)
    {
        waveformFile.read((char *) &data, sizeof(uint16_t) * colums);
        //First entry is the channel
        waveform->lowPower.push_back(data[0]);

        //Start with 1 since the first entry was channelinfo
        for(int j = 1; j < colums; j++)
        {
            if(data[j] == 0 || j == colums - 1)
            {
                if(data[j] != 0 )
                {
                    waveform->waveformSamples.push_back(data[j]);
                    waveform->waveformIndices.push_back(waveform->waveformIndices.back() + j);
                }
                else
                {
                    waveform->waveformIndices.push_back(waveform->waveformIndices.back() + j - 1);
                }
                break;
            }
            waveform->waveformSamples.push_back(data[j]);
        }
    }
    waveform->maxBucketSize = colums - 1;

    //find correpsonding item
    std::map<QString, ModelBridgePtr> possibleScans;
    QStringList scanNames;
    QTreeWidgetItemIterator itu(treeWidget);
    while (*itu)
    {
        QTreeWidgetItem* item = *itu;

        if ( item->type() == LVRPointCloudItemType)
        {
            auto citem = static_cast<LVRPointCloudItem*>(*itu);
            if (citem->getNumPoints() == waveform->lowPower.size())
            {
                QString key = item->parent()->parent()->text(0) + "\\" + item->parent()->text(0);
                auto scanItem = static_cast<LVRModelItem*>(item->parent());
                possibleScans[key] = scanItem->getModelBridge();
                scanNames << key; 
            }
        }
        itu++;
    }
    if(scanNames.size() == 0)
    {
        QMessageBox noPCDialog;
        noPCDialog.setText("No pointcloud of the same size was found. Abort");
        noPCDialog.setStandardButtons(QMessageBox::Ok);
        noPCDialog.setIcon(QMessageBox::Warning);
        int returnValue = noPCDialog.exec();
        return;
    }
    if(scanNames.size() == 1)
    {
        possibleScans[scanNames[0]]->setWaveform(waveform);
        return;
    }
    bool ok;

    QString choosenItem = QInputDialog::getItem(this, tr("Choose corresponding pointcloud"),

                                                     tr("Choose corresponding pointcloud:"), scanNames, 0, false, &ok);
    if(!ok)
    {
        return;
    }
    possibleScans[choosenItem]->setWaveform(waveform);

}
LVRLabeledScanProjectEditMarkItem* LVRMainWindow::checkForScanProject()
{
    for(int i = 0; i < treeWidget->topLevelItemCount(); i++)
    {
        auto topitem = treeWidget->topLevelItem(i);

        if(topitem->type() != LVRLabeledScanProjectEditMarkItemType)
        {
            if(topitem->type() == LVRModelItemType)
            {
                QMessageBox::StandardButton info =  QMessageBox::question(this, "Tranform Pointcloud?", "The found Pointcloud is not park of a ScanProject. The requested operation requires a ScanProject. Shall the Pointcloud be transfromed into a ScanProject?",QMessageBox::Yes|QMessageBox::No);
                if (info != QMessageBox::Yes) 
                {
                    return nullptr;
                }
                auto mitem = static_cast<LVRModelItem*>(topitem);
                LabeledScanProjectEditMarkBridgePtr transfer = LabeledScanProjectEditMarkBridgePtr(new LVRLabeledScanProjectEditMarkBridge(mitem->getModelBridge()));
                LVRLabeledScanProjectEditMarkItem* item = new LVRLabeledScanProjectEditMarkItem(transfer, "LabelScanProject");
                treeWidget->addTopLevelItem(item);
                delete treeWidget->takeTopLevelItem(i);
                return item;
            }
        }
        if(topitem->type() == LVRLabeledScanProjectEditMarkItemType)
        {
            return static_cast<LVRLabeledScanProjectEditMarkItem*>(topitem);
        }
    }
    return nullptr;
}

void LVRMainWindow::exportScanProject()
{
    QString hdfString("HDF5 (*.hdf5");
    QString intermediaString("Intermedia");
    QStringList filters;
    filters << hdfString << intermediaString;
    QFileDialog dialog(this, tr("Export ScanProject As..."), "", tr("HDF5 (*.hdf5);; Intermedia (*.intermedia)"));
    dialog.setNameFilters(filters);
    dialog.setAcceptMode(QFileDialog::AcceptSave);
    if (dialog.exec() != QDialog::Accepted)
    {
        //No valid inut return
        return;
    }

    for(int i = 0; i < treeWidget->topLevelItemCount(); i++)
    {
        QTreeWidgetItem* topItem = treeWidget->topLevelItem(i);
        LabeledScanProjectEditMarkPtr labeledScanProject;
        QString fileName = dialog.selectedFiles()[0];

        if (topItem->type() == LVRLabeledScanProjectEditMarkItemType)
        {
            LVRLabeledScanProjectEditMarkItem *item = static_cast<LVRLabeledScanProjectEditMarkItem *>(topItem);
            /*
            LVRLabeledScanProjectEditMarkBridge transfer(item->getLabeledScanProjectEditMarkBridge()->getScanProjectBridgePtr()->getScanPositions()[0]->getModels()[0]);
            labeledScanProject = transfer.getLabeledScanProjectEditMark();*/
            labeledScanProject = item->getLabeledScanProjectEditMarkBridge()->getLabeledScanProjectEditMark();
        }
        else if(topItem->type() == LVRModelItemType)
        {
            LVRModelItem *item = static_cast<LVRModelItem *>(topItem);
            LVRLabeledScanProjectEditMarkBridge transfer(item->getModelBridge());
            labeledScanProject = transfer.getLabeledScanProjectEditMark();

            //check if Labels exists and add them
            if(labelTreeWidget->topLevelItemCount() > 0)
            {
                labeledScanProject->labelRoot = labelTreeWidget->getLabelRoot(); 
            }

        }
        else
        {
            continue;
        }
        
        //store Project
        if (dialog.selectedNameFilter() == hdfString)
        {
            //as HDF5
            LabelHDF5SchemaPtr hdf5Schema(new LabelScanProjectSchemaHDF5V2);
            HDF5KernelPtr hdf5Kernel(new HDF5Kernel(fileName.toStdString() + ".h5"));
            LabelHDF5IO h5IO(hdf5Kernel, hdf5Schema);

            h5IO.saveLabelScanProject(labeledScanProject);

        }else
        {

            //Intermedia
            std::string tmp = fileName.toStdString();
            DirectorySchemaPtr hyperlibSchema(new ScanProjectSchemaHyperlib(tmp));
            DirectoryKernelPtr dirKernelPtr(new DirectoryKernel(fileName.toStdString()));
            DirectoryIO dirIO(dirKernelPtr, hyperlibSchema);
        
            dirIO.saveScanProject(labeledScanProject->editMarkProject->project);
        }
    }

}
} /* namespace lvr2 */
