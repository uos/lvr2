#####################################################################################
# Avoid the name collision with boost signals
#####################################################################################

ADD_DEFINITIONS(-DQT_NO_KEYWORDS )
set(CMAKE_AUTOMOC ON)

#####################################################################################
# Set source files
#####################################################################################

set(LVR_VIEWER_SOURCES
    app/LVRMain.cpp
    app/LVRMainWindow.cpp
    app/LVRTreeWidgetHelper.cpp
    vtkBridge/LVRModelBridge.cpp
    vtkBridge/LVRPointBufferBridge.cpp
    vtkBridge/LVRMeshBufferBridge.cpp
    vtkBridge/LVRPickingInteractor.cpp
    vtkBridge/LVRVtkArrow.cpp
    widgets/LVRModelItem.cpp
    widgets/LVRPointCloudItem.cpp
    widgets/LVRMeshItem.cpp
    widgets/LVRPoseItem.cpp
    widgets/LVRPickItem.cpp
    widgets/LVRTextureMeshItem.cpp
    widgets/LVRRecordedFrameItem.cpp
    widgets/LVRRenameDialog.cpp
    widgets/LVRAnimationDialog.cpp
    widgets/LVRTransformationDialog.cpp
    widgets/LVRCorrespondanceDialog.cpp
    widgets/LVRReconstructionEstimateNormalsDialog.cpp
    widgets/LVRReconstructionMarchingCubesDialog.cpp
    widgets/LVRReconstructionExtendedMarchingCubesDialog.cpp
    widgets/LVROptimizationPlanarOptimizationDialog.cpp
    widgets/LVROptimizationRemoveArtifactsDialog.cpp
    widgets/LVRFilteringRemoveOutliersDialog.cpp
    widgets/LVRFilteringMLSProjectionDialog.cpp
    widgets/LVRBackgroundDialog.cpp
)

#####################################################################################
# Set mocs
#####################################################################################

set(LVR_VIEWER_MOCS
    app/LVRMainWindow.hpp
    vtkBridge/LVRPickingInteractor.hpp
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
    widgets/LVRCorrespondanceDialog.hpp
    widgets/LVRBackgroundDialog.hpp
)

#####################################################################################
# Set uis
#####################################################################################

set(LVR_VIEWER_UIS
   ui/LVRMainWindowUI.ui
   ui/LVRRenameDialogUI.ui
   ui/LVRAnimationDialogUI.ui
   ui/LVRTransformationDialogUI.ui
   ui/LVRRegistrationPickCorrespondancesDialogUI.ui
   ui/LVRAboutDialogUI.ui
   ui/LVRReconstructionMarchingCubesDialogUI.ui
   ui/LVRReconstructionExtendedMarchingCubesDialogUI.ui
   ui/LVRReconstructionEstimateNormalsDialogUI.ui
   ui/LVROptimizationPlanarOptimizationDialogUI.ui
   ui/LVROptimizationRemoveArtifactsDialogUI.ui
   ui/LVRFilteringRemoveOutliersDialogUI.ui
   ui/LVRFilteringMLSProjectionDialogUI.ui
   ui/LVRBackgroundDialogUI.ui
)

#####################################################################################
# Set resources
#####################################################################################

set(LVR_VIEWER_RESOURCES
   res/viewer.qrc
)

#####################################################################################
# Setup Visual Studio Project
#####################################################################################

if(MSVC)
  include(LVRVSSetup.txt)
endif()

#####################################################################################
# Generate ui and resource headers
#####################################################################################

qt5_wrap_ui(LVR_VIEWER_UI_HDRS ${LVR_VIEWER_UIS})
qt5_add_resources(LVR_VIEWER_RES ${LVR_VIEWER_RESOURCES})

add_definitions(${QT_DEFINITIONS})

#####################################################################################
# Add current binary dir to includes to find the generated header files
#####################################################################################

include_directories(${CMAKE_CURRENT_BINARY_DIR})

#####################################################################################
# Setup dependencies
#####################################################################################

set(LVR_VIEWER_DEPENDENCIES
  lvr_static
  lvrrply_static
  lvrlas_static
  lvrslam6d_static
  ${OpenCV_LIBS}
  ${Qt5Widgets_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
  ${Boost_THREAD_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${QVTK_LIBRARY}
  ${VTK_LIBRARIES}
  ${PCL_LIBRARIES}
)

if( UNIX )
  set(LVR_VIEWER_DEPENDENCIES ${LVR_VIEWER_DEPENDENCIES} pthread)
endif( UNIX )

if( ${NABO_FOUND} )
  set(LVR_VIEWER_DEPENDENCIES ${LVR_VIEWER_DEPENDENCIES} ${NABO_LIBRARY})
endif( ${NABO_FOUND} )

#####################################################################################
# Add main executable
#####################################################################################

add_executable(lvr_viewer
  ${LVR_VIEWER_SOURCES}
  ${LVR_VIEWER_RES}
  ${LVR_VIEWER_MOCS}
  ${LVR_VIEWER_UI_HDRS})
target_link_libraries(lvr_viewer ${LVR_VIEWER_DEPENDENCIES})

install(TARGETS lvr_viewer
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})