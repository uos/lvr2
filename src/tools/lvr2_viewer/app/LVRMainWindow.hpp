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
#include <vtkEDLShading.h>
#include <vtkRenderStepsPass.h>
#include <vtkOpenGLRenderer.h>
#include <vtkNew.h>

#include "../widgets/LVRPlotter.hpp"
#include <QtGui>
#include "ui_LVRMainWindowUI.h"
#include "ui_LVRAboutDialogUI.h"
#include "ui_LVRTooltipDialogUI.h"
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
#include "../widgets/LVRHistogram.hpp"
#include "../widgets/LVRScanDataItem.hpp"
#include "../widgets/LVRCamDataItem.hpp"
#include "../widgets/LVRBoundingBoxItem.hpp"

#include "../widgets/LVRPointInfo.hpp"

#include "../vtkBridge/LVRPickingInteractor.hpp"
#include "../vtkBridge/LVRVtkArrow.hpp"

#include <iostream>
#include <iterator>
#include <vector>
#include <set>
#include "boost/format.hpp"
using std::vector;
using std::cout;
using std::endl;

namespace lvr2
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
    /// Shows a Popup Dialog saying that no PointClouds with spectral data are selected
    void showErrorDialog();
    /// Shows a Popup Dialog with the average Intensity per Spectral Channel
    void showHistogramDialog();
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
    void loadPointCloudData();
    void unloadPointCloudData();
    void changePointSize(int pointSize);
    void changeTransparency(int transparencyValue);
    void changeShading(int shader);

    void showImage();
    void setViewToCamera();

    /// Updates all selected LVRPointCloudItems to the desired Spectral. **can take seconds**
    void changeSpectralColor();
    /// Determines if changeSpectralColor() should be called. Updates the m_spectralLineEdit to the value from m_spectralSlider
    void onSpectralSliderChanged(int action = -1);
    /// Updates the m_spectralSlider to the value from m_spectralLineEdit
    void onSpectralLineEditChanged();
    /// Same as onSpectralLineEditChanged(), but triggers changeSpectralView()
    void onSpectralLineEditSubmit();

    /// Updates all selected LVRPointCloudItems to the desired Gradient. **can take seconds**
    void changeGradientColor();
    /// Determines if changeGradientColor() should be called. Updates the m_gradientLineEdit to the value from m_gradientSlider
    void onGradientSliderChanged(int action = -1);
    /// Updates the m_gradientSlider to the value from m_gradientLineEdit
    void onGradientLineEditChanged();
    /// Same as onGradientLineEditChanged(), but triggers changeGradientView()
    void onGradientLineEditSubmit();

    void assertToggles();
    void togglePoints(bool checkboxState);
    void toggleNormals(bool checkboxState);
    void toggleMeshes(bool checkboxState);
    void toggleWireframe(bool checkboxState);
    void toogleEDL(bool checkboxstate);
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

    /// Shows a Popup Dialog with Information about a Point
    void showPointInfoDialog();
    /// Shows the DockerWidget with the preview of the PointInfoDialog
    void showPointPreview(vtkActor* actor, int point);
    /// Changes the Point displayed by the PointPreview
    void updatePointPreview(int pointId, PointBufferPtr points);

    /// Switches between Sliders and Gradients. checked == true => Slider DockWidget enabled
    void updateSpectralSlidersEnabled(bool checked);
    /// Switches between Sliders and Gradients. checked == true => Gradient DockWidget enabled
    void updateSpectralGradientEnabled(bool checked);
    QTreeWidgetItem* addScanData(std::shared_ptr<ScanDataManager> sdm, QTreeWidgetItem *parent);

    LVRModelItem* getModelItem(QTreeWidgetItem* item);
    LVRPointCloudItem* getPointCloudItem(QTreeWidgetItem* item);
    LVRMeshItem* getMeshItem(QTreeWidgetItem* item);
    std::set<LVRModelItem*> getSelectedModelItems();
    std::set<LVRPointCloudItem*> getSelectedPointCloudItems();
    std::set<LVRMeshItem*> getSelectedMeshItems();

protected Q_SLOTS:
    void setModelVisibility(QTreeWidgetItem* treeWidgetItem, int column);
    /// Adjusts all the Sliders, LineEdits and CheckBoxes to the currently selected Items
    void restoreSliders();
    void highlightBoundingBoxes();

Q_SIGNALS:
    void correspondenceDialogOpened();

private:
    void setupQVTK();
    void connectSignalsAndSlots();

    LVRCorrespondanceDialog*                    m_correspondanceDialog;
    std::map<LVRPointCloudItem*, LVRHistogram*> m_histograms;
    LVRPlotter*                                 m_PointPreviewPlotter;
    int                                         m_previewPoint;
    PointBufferPtr                              m_previewPointBuffer;
    QDialog*                                    m_aboutDialog;
    QDialog*                                    m_errorDialog;
    QMessageBox*                                m_incompatibilityBox;
    vtkSmartPointer<vtkRenderer>                m_renderer;
    vtkSmartPointer<vtkRenderWindowInteractor>  m_renderWindowInteractor;
    vtkSmartPointer<vtkCamera>                  m_camera;
    vtkSmartPointer<vtkCameraRepresentation>    m_pathCamera;
    vtkSmartPointer<vtkOrientationMarkerWidget> m_axesWidget;
    vtkSmartPointer<vtkAxesActor>               m_axes;

    QMenu*                                      m_treeParentItemContextMenu;
    QMenu*                                      m_treeChildItemContextMenu;

    // Toolbar item "File"
    QAction*                            m_actionOpen;
    QAction*                            m_actionExport;
    QAction*                            m_actionQuit;
    // Toolbar item "Views"
    QAction*                            m_actionReset_Camera;
    QAction*                            m_actionStore_Current_View;
    QAction*                            m_actionRecall_Stored_View;
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
    QAction*                            m_actionShow_Points;
    QAction*                            m_actionShow_Normals;
    QAction*                            m_actionShow_Mesh;
    QAction*                            m_actionShow_Wireframe;
    QAction*                            m_actionShowBackgroundSettings;
    QAction*                            m_actionShowSpectralSlider;
    QAction*                            m_actionShowSpectralColorGradient;
    QAction*                            m_actionShowSpectralPointPreview;
    QAction*                            m_actionShowSpectralHistogram;
    // Sliders below tree widget
    QSlider*                            m_horizontalSliderPointSize;
    QSlider*                            m_horizontalSliderTransparency;
    // Combo boxes below sliders
    QComboBox*                          m_comboBoxGradient;
    QComboBox*                          m_comboBoxShading;
    // Buttons below combo boxes
    QPushButton*                        m_buttonCameraPathTool;
    QPushButton*                        m_buttonCreateMesh;
    QPushButton*                        m_buttonExportData;
    QPushButton*                        m_buttonTransformModel;
    // Spectral Settings
    QSlider*                            m_spectralSliders[3];
    QCheckBox*                          m_spectralCheckboxes[3];
    QLabel*                             m_spectralLabels[3];
    QLineEdit*                          m_spectralLineEdits[3];
    // Gradient Settings
    QSlider*                            m_gradientSlider;
    QLineEdit*                          m_gradientLineEdit;
    // ContextMenu Items
    QAction*                            m_actionShowColorDialog;
    QAction*                            m_actionRenameModelItem;
    QAction*                            m_actionDeleteModelItem;
    QAction*                            m_actionExportModelTransformed;
    QAction*                            m_actionLoadPointCloudData;
    QAction*                            m_actionUnloadPointCloudData;

    QAction*                            m_actionShowImage;
    QAction*                            m_actionSetViewToCamera;

    LVRPickingInteractor*               m_pickingInteractor;
    LVRTreeWidgetHelper*                m_treeWidgetHelper;

    // EDM Rendering
    vtkSmartPointer<vtkRenderStepsPass> m_basicPasses;
    vtkSmartPointer<vtkEDLShading>      m_edl;


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

} /* namespace lvr2 */

#endif /* MAINWINDOW_HPP_ */
