#####################################################################################
# Avoid the name collision with boost signals
#####################################################################################

ADD_DEFINITIONS(-DQT_NO_KEYWORDS )
set(CMAKE_AUTOMOC ON)

#####################################################################################
# Set source files
#####################################################################################

set(LVR2_VIEWER_SOURCES
    app/LVRMain.cpp
    app/LVRMainWindow.cpp
    app/LVRTreeWidgetHelper.cpp
    app/Options.cpp
    vtkBridge/LVRBoundingBoxBridge.cpp
    vtkBridge/LVRModelBridge.cpp
    vtkBridge/LVRScanProjectBridge.cpp
    vtkBridge/LVRSoilAssistBridge.cpp
    vtkBridge/LVRScanPositionBridge.cpp
    vtkBridge/LVRPointBufferBridge.cpp
    vtkBridge/LVRMeshBufferBridge.cpp
    vtkBridge/LVRLabeledScanProjectEditMarkBridge.cpp
    vtkBridge/LVRLabelBridge.cpp
    vtkBridge/LVRPickingInteractor.cpp
    vtkBridge/LVRInteractorStylePolygonPick.cpp
    vtkBridge/LVRLabelInteractor.cpp
    vtkBridge/LVRVtkArrow.cpp
    vtkBridge/LVRChunkedMeshCuller.cpp
    vtkBridge/LVRChunkedMeshBridge.cpp
    vtkBridge/LVRScanCamBridge.cpp
    vtkBridge/LVRScanImageBridge.cpp
    widgets/LVRScanProjectItem.cpp
    widgets/LVRScanPositionItem.cpp
    widgets/LVRModelItem.cpp
    widgets/LVRPointCloudItem.cpp
    widgets/LVRMeshItem.cpp
    widgets/LVRPoseItem.cpp
    widgets/LVRPickItem.cpp
    widgets/LVRLabelItem.cpp
    widgets/LVRTextureMeshItem.cpp
    widgets/LVRRecordedFrameItem.cpp
    widgets/LVRRenameDialog.cpp
    widgets/LVRAnimationDialog.cpp
    widgets/LVRTransformationDialog.cpp
    widgets/LVRLabelDialog.cpp
    widgets/LVRCorrespondanceDialog.cpp
    widgets/LVRReconstructionEstimateNormalsDialog.cpp
    widgets/LVRReconstructionMarchingCubesDialog.cpp
    widgets/LVRReconstructionExtendedMarchingCubesDialog.cpp
    widgets/LVROptimizationPlanarOptimizationDialog.cpp
    widgets/LVROptimizationRemoveArtifactsDialog.cpp
    widgets/LVRFilteringRemoveOutliersDialog.cpp
    widgets/LVRFilteringMLSProjectionDialog.cpp
    widgets/LVRBackgroundDialog.cpp
    widgets/LVRPointcloudSelectionDialog.cpp
    widgets/LVRPointInfo.cpp
    widgets/LVRPlotter.cpp
    widgets/LVRHistogram.cpp
#    widgets/LVRScanDataItem.cpp
#    widgets/LVRCamDataItem.cpp
#    widgets/LVRCvImageItem.cpp
    widgets/LVRBoundingBoxItem.cpp
    widgets/LVRGraphicsView.cpp
    widgets/LVRLabelTreeWidget.cpp
    widgets/LVRLabelClassTreeItem.cpp
    widgets/LVRLabelInstanceTreeItem.cpp
    widgets/LVRModelTreeWidget.cpp
    widgets/LVRLabeledScanProjectEditMarkItem.cpp
    widgets/LVRScanCamItem.cpp
    widgets/LVRScanImageItem.cpp
    widgets/LVRScanProjectOpenDialog.cpp
    widgets/LVRReductionAlgorithmDialog.cpp
#    util/qttf.cpp
    widgets/LVRCameraModelItem.cpp
    widgets/LVRImageModelItem.cpp
    widgets/LVRExtrinsicsItem.cpp
    widgets/LVRMeshOpenDialog.cpp
)

#if(DEFINED VTK_MAJOR_VERSION AND VTK_MAJOR_VERSION VERSION_EQUAL "8")
#  list(APPEND LVR2_VIEWER_SOURCES
#    vtkBridge/PreloadOpenGLPolyDataMapper.cpp
#)
#endif()

#####################################################################################
# Set mocs
#####################################################################################

set(LVR2_VIEWER_MOCS
    app/LVRMainWindow.hpp
    vtkBridge/LVRPickingInteractor.hpp
    vtkBridge/LVRInteractorStylePolygonPick.hpp
    vtkBridge/LVRLabelInteractor.hpp
    widgets/LVRRenameDialog.hpp
    widgets/LVRAnimationDialog.hpp
    widgets/LVRTransformationDialog.hpp
    widgets/LVRReconstructionEstimateNormalsDialog.hpp
    widgets/LVRReconstructionMarchingCubesDialog.hpp
    widgets/LVRReconstructionExtendedMarchingCubesDialog.hpp
    widgets/LVROptimizationPlanarOptimizationDialog.hpp
    widgets/LVROptimizationRemoveArtifactsDialog.hpp
    widgets/LVRFilteringRemoveOutliersDialog.hpp
    widgets/LVRFilteringMLSProjectionDialog.hpp
    widgets/LVRLabelDialog.hpp
    widgets/LVRCorrespondanceDialog.hpp
    widgets/LVRBackgroundDialog.hpp
    widgets/LVRPointcloudSelectionDialog.hpp
    widgets/LVRPointInfo.hpp
    widgets/LVRPlotter.hpp
    widgets/LVRHistogram.hpp
    widgets/LVRScanProjectOpenDialog.hpp
    widgets/LVRReductionAlgorithmDialog.hpp
    widgets/LVRMeshOpenDialog.hpp
)

#####################################################################################
# Set uis
#####################################################################################


# LEGACY
set(LVR2_VIEWER_UIS
    ui/LVRMainWindowUI.ui
)

list(APPEND LVR2_VIEWER_UIS
   ui/LVRRenameDialogUI.ui
   ui/LVRAnimationDialogUI.ui
   ui/LVRTransformationDialogUI.ui
   ui/LVRRegistrationPickCorrespondancesDialogUI.ui
   ui/LVRLabelDialogUI.ui
   ui/LVRAboutDialogUI.ui
   ui/LVRReconstructionMarchingCubesDialogUI.ui
   ui/LVRReconstructionExtendedMarchingCubesDialogUI.ui
   ui/LVRReconstructionEstimateNormalsDialogUI.ui
   ui/LVROptimizationPlanarOptimizationDialogUI.ui
   ui/LVROptimizationRemoveArtifactsDialogUI.ui
   ui/LVRFilteringRemoveOutliersDialogUI.ui
   ui/LVRFilteringMLSProjectionDialogUI.ui
   ui/LVRBackgroundDialogUI.ui
   ui/LVRPointInfo.ui
   ui/LVRTooltipDialogUI.ui
   ui/LVRHistogram.ui
   ui/LVRScanProjectOpenDialogUI.ui
   ui/LVRReductionAlgorithmDialogUI.ui
   ui/LVRMeshOpenDialogUI.ui
)

#####################################################################################
# Set resources
#####################################################################################

set(LVR2_VIEWER_RESOURCES
   res/viewer.qrc
)

#####################################################################################
# Generate ui and resource headers
#####################################################################################

qt5_wrap_ui(LVR2_VIEWER_UI_HDRS ${LVR2_VIEWER_UIS})
qt5_add_resources(LVR2_VIEWER_RES ${LVR2_VIEWER_RESOURCES})

add_definitions(${QT_DEFINITIONS})

#####################################################################################
# Add current binary dir to includes to find the generated header files
#####################################################################################

include_directories(${CMAKE_CURRENT_BINARY_DIR} widgets)

#####################################################################################
# Setup dependencies
#####################################################################################

set(LVR2_VIEWER_DEPENDENCIES
  lvr2_static
  lvr2rply_static
  lvr2las_static
  ${LVR2_LIB_DEPENDENCIES}
  ${Qt5Widgets_LIBRARIES}
  ${VTK_LIBRARIES}
  ${PCL_LIBRARIES}
)

if(DEFINED VTK_MAJOR_VERSION AND VTK_MAJOR_VERSION VERSION_LESS "8")
  set(LVR2_VIEWER_DEPENDENCIES ${LVR2_VIEWER_DEPENDENCIES} ${QVTK_LIBRARY})
endif()

if( UNIX )
  set(LVR2_VIEWER_DEPENDENCIES ${LVR2_VIEWER_DEPENDENCIES} pthread)
endif( UNIX )

#####################################################################################
# Add main executable
#####################################################################################
add_executable(lvr2_viewer
  ${LVR2_VIEWER_SOURCES}
  ${LVR2_VIEWER_RES}
  ${LVR2_VIEWER_MOCS}
  ${LVR2_VIEWER_UI_HDRS})

target_link_libraries(lvr2_viewer ${LVR2_VIEWER_DEPENDENCIES})
target_precompile_headers(lvr2_viewer PUBLIC ${LVR2_PCHS})

if(CUDA_FOUND)
        target_link_libraries(lvr2_viewer lvr2cuda)
        target_compile_definitions(lvr2_viewer PRIVATE CUDA_FOUND=1)
endif()
if(OPENCL_FOUND)
        target_compile_definitions(lvr2_viewer PRIVATE OPENCL_FOUND=1)
endif()

install(TARGETS lvr2_viewer
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
