# Objects that use Qt's signals and slots

ADD_DEFINITIONS(-DQT_NO_KEYWORDS )

# Main Sources
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
    widgets/LVRTransformationDialog.cpp
    widgets/LVRCorrespondanceDialog.cpp
    widgets/LVRReconstructionMarchingCubesDialog.cpp
    widgets/LVROptimizationPlanarOptimizationDialog.cpp
    widgets/LVROptimizationRemoveArtifactsDialog.cpp
    widgets/LVRFilteringRemoveOutliersDialog.cpp
    widgets/LVRFilteringMLSProjectionDialog.cpp
)


set(LVR_VIEWER_MOCS 
    app/LVRMainWindow.hpp
    vtkBridge/LVRPickingInteractor.hpp
    widgets/LVRTransformationDialog.hpp
    widgets/LVRReconstructionMarchingCubesDialog.hpp
    widgets/LVROptimizationPlanarOptimizationDialog.hpp
    widgets/LVROptimizationRemoveArtifactsDialog.hpp
    widgets/LVRFilteringRemoveOutliersDialog.hpp
    widgets/LVRFilteringMLSProjectionDialog.hpp
    widgets/LVRCorrespondanceDialog.hpp
)

# Dynamically generated GUI object's from
# QtDesigner 
set(LVR_VIEWER_UIS
   ui/LVRMainWindowUI.ui
   ui/LVRTransformationDialogUI.ui
   ui/LVRRegistrationPickCorrespondancesDialogUI.ui
   ui/LVRAboutDialogUI.ui
   ui/LVRReconstructionMarchingCubesDialogUI.ui
   ui/LVROptimizationPlanarOptimizationDialogUI.ui
   ui/LVROptimizationRemoveArtifactsDialogUI.ui
   ui/LVRFilteringRemoveOutliersDialogUI.ui
   ui/LVRFilteringMLSProjectionDialogUI.ui
)

set(LVR_VIEWER_RESOURCES
   res/viewer.qrc
)

# Generate mocs and header files for qt objects
# and forms
qt_wrap_cpp(lvr_viewer LVR_VIEWER_MOC_SRC ${LVR_VIEWER_MOCS})
qt_wrap_ui(lvr_viewer LVR_VIEWER_UI_HDRS LVR_VIEWER_UI_SRC ${LVR_VIEWER_UIS})
qt4_add_resources(LVR_VIEWER_RES ${LVR_VIEWER_RESOURCES})

add_definitions(${QT_DEFINITIONS})

# Add current binary dir to includes to find
# the generated header files
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LAS_VEGAS_SOURCE_DIR}/include/liblvr)

# Create a new targer
add_executable(lvr_viewer ${LVR_VIEWER_SOURCES} ${LVR_VIEWER_RES} ${LVR_VIEWER_MOC_SRC} ${LVR_VIEWER_UI_HDRS})

# Add needed libraries
target_link_libraries(lvr_viewer ${QT_QTCORE_LIBRARIES})
target_link_libraries(lvr_viewer ${QT_QTGUI_LIBRARIES})
target_link_libraries(lvr_viewer ${QT_QTOPENGL_LIBRARIES})
target_link_libraries(lvr_viewer ${QT_QTXML_LIBRARY})
target_link_libraries(lvr_viewer lvr)
target_link_libraries(lvr_viewer ${OPENGL_LIBRARIES})
target_link_libraries(lvr_viewer ${GLUT_LIBRARIES})
target_link_libraries(lvr_viewer ${QGLVIEWER_LIBRARY_RELEASE})
target_link_libraries(lvr_viewer GLU)
target_link_libraries(lvr_viewer ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(lvr_viewer pthread)
target_link_libraries(lvr_viewer GLU)
target_link_libraries(lvr_viewer ${PCL_LIBRARIES})
target_link_libraries(lvr_viewer GLU QVTK)

INSTALL(TARGETS lvr_viewer
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
