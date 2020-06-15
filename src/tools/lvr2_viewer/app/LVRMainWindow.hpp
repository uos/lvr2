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
#include <vtkOpenGLRenderer.h>
#include <vtkNew.h>

#include <vtkCullerCollection.h>

// EDL shading is only available in new vtk versions
#ifdef LVR2_USE_VTK_GE_7_1
#include <vtkEDLShading.h>
#include <vtkRenderStepsPass.h>
#endif

#include <QtGui>
#ifdef LVR2_USE_VTK8
    #include "ui_LVRMainWindowQVTKOGLUI.h"
#else
    #include "ui_LVRMainWindowUI.h"
#endif
#include "ui_LVRAboutDialogUI.h"
#include "ui_LVRTooltipDialogUI.h"

#include "LVRTreeWidgetHelper.hpp"

#include "../widgets/LVRPlotter.hpp"
#include "../vtkBridge/LVRModelBridge.hpp"
#include "../widgets/LVRModelItem.hpp"
#include "../widgets/LVRPointCloudItem.hpp"
#include "../widgets/LVRMeshItem.hpp"
#include "../widgets/LVRItemTypes.hpp"
#include "../widgets/LVRRenameDialog.hpp"
#include "../widgets/LVRAnimationDialog.hpp"
#include "../widgets/LVRTransformationDialog.hpp"
#include "../widgets/LVRCorrespondanceDialog.hpp"
#include "../widgets/LVRLabelDialog.hpp"
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
#include "../vtkBridge/LVRLabelInteractor.hpp"
#include "../vtkBridge/LVRVtkArrow.hpp"

#include <iostream>
#include <iterator>
#include <vector>
#include <set>
#include <boost/format.hpp>

#include "../vtkBridge/LVRChunkedMeshBridge.hpp"
#include "../vtkBridge/LVRChunkedMeshCuller.hpp"

#define LABEL_NAME_COLUMN 0
#define LABELED_POINT_COLUMN 1
#define LABEL_VISIBLE_COLUMN 2
#define LABEL_ID_COLUMN 3



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
    std::mutex display_mutex;

public Q_SLOTS:
    void updateDisplayLists(actorMap lowRes, actorMap highRes);
            
            //std::unordered_map<size_t, vtkSmartPointer<MeshChunkActor> > lowResActors,
            //                std::unordered_map<size_t, vtkSmartPointer<MeshChunkActor> > highResActors);


    void comboBoxIndexChanged(int index);
    void addNewInstance(QTreeWidgetItem *);
    void loadModel();
    void loadModels(const QStringList& filenames);
    void loadChunkedMesh();
    void loadChunkedMesh(const QStringList& filenames, std::vector<std::string> layers, int cacheSize, float highResDistance);
    void manualICP();
    void manualLabeling();
    void changePicker(bool labeling);
    void showLabelTreeContextMenu(const QPoint&);
    void updatePointCount(const uint16_t, const int);

    void cellSelected(QTreeWidgetItem* item, int column);
    void addLabelClass();

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
    void copyModelItem();
    void pasteModelItem();
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
    QTreeWidgetItem* addScans(std::shared_ptr<ScanDataManager> sdm, QTreeWidgetItem *parent);

    LVRModelItem* getModelItem(QTreeWidgetItem* item);
    LVRPointCloudItem* getPointCloudItem(QTreeWidgetItem* item);
    QList<LVRPointCloudItem*> getPointCloudItems(QList<QTreeWidgetItem*> items);
    LVRMeshItem* getMeshItem(QTreeWidgetItem* item);
    std::set<LVRModelItem*> getSelectedModelItems();
    std::set<LVRPointCloudItem*> getSelectedPointCloudItems();
    std::set<LVRMeshItem*> getSelectedMeshItems();
    void exportLabels();

protected Q_SLOTS:
    void setModelVisibility(QTreeWidgetItem* treeWidgetItem, int column);
    /// Adjusts all the Sliders, LineEdits and CheckBoxes to the currently selected Items
    void restoreSliders();
    void highlightBoundingBoxes();

    void visibilityChanged(QTreeWidgetItem*, int);
    void loadLabels();

Q_SIGNALS:
    void labelChanged(uint16_t);
    void correspondenceDialogOpened();
    void labelAdded(QTreeWidgetItem*);
    void hidePoints(int, bool);
    void labelLoaded(int, std::vector<int>);

private:
    void setupQVTK();
    void connectSignalsAndSlots();
    LVRModelItem* loadModelItem(QString name);
    bool childNameExists(QTreeWidgetItem* item, const QString& name);
    QString increaseFilename(QString filename);

    QList<QTreeWidgetItem*>                     m_items_copied;
    LVRCorrespondanceDialog*                    m_correspondanceDialog;
    LVRLabelDialog*                   		m_labelDialog;
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

    ChunkedMeshBridgePtr m_chunkBridge;
//    vtkSmartPointer<ChunkedMeshCuller> m_chunkCuller;
    ChunkedMeshCuller* m_chunkCuller;
    QMenu*                                      m_treeParentItemContextMenu;
    QMenu*                                      m_treeChildItemContextMenu;

    QMenu*                                      m_labelTreeParentItemContextMenu;
    QMenu*                                      m_labelTreeChildItemContextMenu;
    // Toolbar item "File"
    QAction*                            m_actionOpen;
    QAction*                            m_actionOpenChunkedMesh;
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
    // Toolbar items "Labeling"
    QAction* 				m_actionStart_labeling;
    QAction* 				m_actionStop_labeling;
    QAction* 				m_actionExtract_labeling;
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
    QAction*                            m_actionCopyModelItem;
    QAction*                            m_actionPasteModelItem;
    QAction*                            m_actionRenameModelItem;
    QAction*                            m_actionDeleteModelItem;
    QAction*                            m_actionExportModelTransformed;
    QAction*                            m_actionLoadPointCloudData;
    QAction*                            m_actionUnloadPointCloudData;

    QAction*                            m_actionShowImage;
    QAction*                            m_actionSetViewToCamera;
    
    //Label
    QAction*                            m_actionAddLabelClass;
    QAction*                            m_actionAddNewInstance;
    QAction*                            m_actionRemoveInstance;

    LVRPickingInteractor*               m_pickingInteractor;
    LVRLabelInteractorStyle*		m_labelInteractor; 
    LVRTreeWidgetHelper*                m_treeWidgetHelper;


    // EDM Rendering
#ifdef LVR2_USE_VTK_GE_7_1
    vtkSmartPointer<vtkRenderStepsPass> m_basicPasses;
    vtkSmartPointer<vtkEDLShading>      m_edl;
#endif

    bool m_labeling = false;
    int m_id = 1;
    static const string UNKNOWNNAME;
    QTreeWidgetItem* m_selectedLabelItem;

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
