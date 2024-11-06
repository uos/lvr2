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
    vtkBridge/LVRPointBufferBridge.cpp
    vtkBridge/LVRMeshBufferBridge.cpp
    vtkBridge/LVRPickingInteractor.cpp
    vtkBridge/LVRInteractorStylePolygonPick.cpp
    vtkBridge/LVRLabelInteractor.cpp
    vtkBridge/LVRVtkArrow.cpp
    vtkBridge/LVRChunkedMeshCuller.cpp
    vtkBridge/LVRChunkedMeshBridge.cpp
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
    widgets/LVRPointInfo.cpp
    widgets/LVRPlotter.cpp
    widgets/LVRHistogram.cpp
    widgets/LVRScanDataItem.cpp
    widgets/LVRCamDataItem.cpp
    widgets/LVRCvImageItem.cpp
    widgets/LVRBoundingBoxItem.cpp
    widgets/LVRGraphicsView.cpp
    util/qttf.cpp
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
    widgets/LVRPointInfo.hpp
    widgets/LVRPlotter.hpp
    widgets/LVRHistogram.hpp
)

#####################################################################################
# Set uis
#####################################################################################

if(DEFINED VTK_MAJOR_VERSION AND VTK_MAJOR_VERSION VERSION_EQUAL "8")
    # newer vtk versions
    set(LVR2_VIEWER_UIS
        ui/LVRMainWindowQVTKOGLUI.ui)
else()
    # LEGACY
    set(LVR2_VIEWER_UIS
        ui/LVRMainWindowUI.ui
    )
endif()

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
)

#####################################################################################
# Set resources
#####################################################################################

set(LVR2_VIEWER_RESOURCES
   res/viewer.qrc
)

#####################################################################################
# Setup Visual Studio Project
#####################################################################################

if(MSVC)
  include(LVR2VSSetup.txt)
endif()

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
  lvr2slam6d_static
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

if( ${NABO_FOUND} )
  set(LVR2_VIEWER_DEPENDENCIES ${LVR2_VIEWER_DEPENDENCIES} ${NABO_LIBRARY})
endif( ${NABO_FOUND} )

#####################################################################################
# Add main executable
#####################################################################################

add_executable(lvr2_viewer
  ${LVR2_VIEWER_SOURCES}
  ${LVR2_VIEWER_RES}
  ${LVR2_VIEWER_MOCS}
  ${LVR2_VIEWER_UI_HDRS})

target_link_libraries(lvr2_viewer 
  ${LVR2_VIEWER_DEPENDENCIES}
  # VTK::GUISupportQt
  ${VTK_LIBRARIES}
  )

target_include_directories(lvr2_viewer PUBLIC ${VTK_INCLUDE_DIRS})

target_include_directories(lvr2_viewer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ext/QVTKOpenGLWidget>
  $<INSTALL_INTERFACE:ext/QVTKOpenGLWidget>  # <prefix>/ext/QVTKOpenGLWidget
)

# target_compile_definitions(lvr2_viewer ${VTK_DEFINITIONS})

message(WARNING "QVTK: ${QVTK_LIBRARY}, ${VTK_}" )

if(CUDA_FOUND)
        target_link_libraries(lvr2_viewer lvr2cuda)
        target_compile_definitions(lvr2_viewer PRIVATE CUDA_FOUND=1)
endif()
if(OPENCL_FOUND)
        target_compile_definitions(lvr2_viewer PRIVATE OPENCL_FOUND=1)
endif()

install(TARGETS lvr2_viewer
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})




  # vtkChartsCore;vtkCommonColor;vtkCommonCore;vtksys;vtkCommonDataModel;vtkCommonMath;vtkCommonMisc;vtkCommonSystem;vtkCommonTransforms;vtkCommonExecutionModel;vtkFiltersGeneral;vtkCommonComputationalGeometry;vtkFiltersCore;vtkInfovisCore;vtkFiltersExtraction;vtkFiltersStatistics;vtkImagingFourier;vtkImagingCore;vtkalglib;vtkRenderingContext2D;vtkRenderingCore;vtkFiltersGeometry;vtkFiltersSources;vtkRenderingFreeType;/usr/lib/x86_64-linux-gnu/libfreetype.so;/usr/lib/x86_64-linux-gnu/libz.so;vtkDICOMParser;vtkDomainsChemistry;vtkIOLegacy;vtkIOCore;vtkIOXMLParser;/usr/lib/x86_64-linux-gnu/libexpat.so;vtkDomainsChemistryOpenGL2;vtkRenderingOpenGL2;vtkIOImage;vtkmetaio;/usr/lib/x86_64-linux-gnu/libjpeg.so;/usr/lib/x86_64-linux-gnu/libpng.so;/usr/lib/x86_64-linux-gnu/libtiff.so;vtkFiltersAMR;vtkIOXML;vtkParallelCore;vtkFiltersFlowPaths;vtkFiltersGeneric;vtkFiltersHybrid;vtkImagingSources;vtkFiltersHyperTree;vtkFiltersImaging;vtkImagingGeneral;vtkFiltersModeling;vtkFiltersParallel;vtkFiltersParallelDIY2;vtkParallelMPI;vtkFiltersParallelFlowPaths;vtkFiltersParallelGeometry;vtkFiltersParallelImaging;vtkFiltersParallelMPI;vtkFiltersParallelStatistics;vtkFiltersPoints;vtkFiltersProgrammable;vtkFiltersPython;/usr/lib/x86_64-linux-gnu/libpython3.8.so;vtkWrappingPythonCore;vtkWrappingTools;vtkFiltersReebGraph;vtkFiltersSMP;vtkFiltersSelection;vtkFiltersTexture;vtkFiltersVerdict;verdict;vtkGUISupportQt;vtkInteractionStyle;vtkGUISupportQtSQL;vtkIOSQL;sqlite3;vtkGeovisCore;vtkInfovisLayout;vtkImagingHybrid;vtkInteractionWidgets;vtkImagingColor;vtkRenderingAnnotation;vtkRenderingVolume;vtkViewsCore;vtkproj4;vtkIOAMR;/usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5.so;/usr/lib/x86_64-linux-gnu/libsz.so;/usr/lib/x86_64-linux-gnu/libdl.so;/usr/lib/x86_64-linux-gnu/libm.so;/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so;/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so;/usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5_hl.so;vtkIOEnSight;vtkIOExodus;vtkexoIIc;/usr/lib/x86_64-linux-gnu/libnetcdf_c++.so;/usr/lib/x86_64-linux-gnu/libnetcdf.so;vtkIOExport;vtkRenderingGL2PSOpenGL2;/usr/lib/x86_64-linux-gnu/libgl2ps.so;vtkIOFFMPEG;vtkIOMovie;/usr/lib/x86_64-linux-gnu/libtheoraenc.so;/usr/lib/x86_64-linux-gnu/libtheoradec.so;/usr/lib/x86_64-linux-gnu/libogg.so;vtkIOGDAL;vtkIOGeoJSON;/usr/lib/x86_64-linux-gnu/libjsoncpp.so;vtkIOGeometry;vtkIOImport;vtkIOInfovis;/usr/lib/x86_64-linux-gnu/libxml2.so;vtkIOLSDyna;vtkIOMINC;vtkIOMPIImage;vtkIOMPIParallel;vtkIOParallel;vtkIONetCDF;vtkIOMySQL;vtkIOODBC;vtkIOPLY;vtkIOParallelExodus;vtkIOParallelLSDyna;vtkIOParallelNetCDF;vtkIOParallelXML;vtkIOPostgreSQL;vtkIOTecplotTable;vtkIOVPIC;VPIC;vtkIOVideo;vtkIOXdmf2;vtkxdmf2;vtkImagingMath;vtkImagingMorphological;vtkImagingStatistics;vtkImagingStencil;vtkInfovisBoostGraphAlgorithms;vtkInteractionImage;vtkLocalExample;vtkParallelMPI4Py;vtkPythonInterpreter;vtkRenderingContextOpenGL2;vtkRenderingExternal;vtkRenderingFreeTypeFontConfig;vtkRenderingImage;vtkRenderingLICOpenGL2;vtkRenderingLOD;vtkRenderingLabel;vtkRenderingMatplotlib;vtkRenderingParallel;vtkRenderingParallelLIC;vtkRenderingSceneGraph;vtkRenderingVolumeAMR;vtkRenderingVolumeOpenGL2;vtkTestingGenericBridge;vtkTestingIOSQL;vtkTestingRendering;vtkViewsContext2D;vtkViewsGeovis;vtkViewsInfovis;vtkWrappingJava
