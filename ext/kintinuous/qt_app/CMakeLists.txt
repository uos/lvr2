#####################################################################################
# Avoid the name collision with boost signals
#####################################################################################

ADD_DEFINITIONS(-DQT_NO_KEYWORDS )

include_directories(../kfusion/include)

#####################################################################################
# Set source files
#####################################################################################

set(KINFU_VIEWER_SOURCES
    MainWindow.cpp
    Main.cpp
    MeshUpdateThread.cpp
)

#####################################################################################
# Set mocs
#####################################################################################

set(KINFU_VIEWER_MOCS 
    MainWindow.hpp
    MeshUpdateThread.hpp
)

#####################################################################################
# Set uis
#####################################################################################

set(KINFU_VIEWER_UIS
   KinfuMainWindow.ui
)

#####################################################################################
# Set resources
#####################################################################################

set(KINFU_VIEWER_RESOURCES
   
)

#####################################################################################
# Setup Visual Studio Project
#####################################################################################

if(MSVC)
include(LVRVSSetup.txt)
endif()



set( QT_USE_QTOPENGL TRUE )
set( QT_USE_XML TRUE)
find_package( Qt4 )
if( QT4_FOUND )
    include_directories(${QT_INCLUDE_DIR})
    include_directories(${QT_QTGUI_INCLUDE_DIR})
    include_directories(${QT_QTOPENGL_INCLUDE_DIR})
    include_directories(${QT_QTCORE_INCLUDE_DIR})
    include_directories(${QT_QTXML_INCLUDE_DIR})
endif(QT4_FOUND)




#####################################################################################
# Generate mocs, ui and resource headers
#####################################################################################

qt_wrap_cpp(kinfu_viewer KINFU_VIEWER_MOC_SRC ${KINFU_VIEWER_MOCS})
qt_wrap_ui(kinfu_viewer KINFU_VIEWER_UI_HDRS KINFU_VIEWER_UI_SRC ${KINFU_VIEWER_UIS})
#qt4_add_resources(LVR2_VIEWER_RES ${LVR2_VIEWER_RESOURCES})

add_definitions(${QT_DEFINITIONS})

#####################################################################################
# Add current binary dir to includes to find the generated header files
#####################################################################################

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LAS_VEGAS_SOURCE_DIR}/include/liblvr)

#####################################################################################
# Setup dependencies
#####################################################################################

set(KINFU_VIEWER_DEPENDENCIES
    kfusion
	lvr_static
	lvrrply_static
	lvrlas_static
	lvrslam6d_static
	${OpenCV_LIBS}
	${QT_QTCORE_LIBRARIES} 
	${QT_QTGUI_LIBRARIES}  
	${QT_QTOPENGL_LIBRARIES} 
	${QT_QTXML_LIBRARY}  
	${OPENGL_LIBRARY} 
	${GLUT_LIBRARIES}  
	${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} 
	${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
	${QVTK_LIBRARY}
	${VTK_LIBRARIES}
	${OPENNI2_LIBRARIES} 
)

if( UNIX )
	set(KINFU_VIEWER_DEPENDENCIES ${KINFU_VIEWER_DEPENDENCIES} pthread)
endif( UNIX ) 

#####################################################################################
# Add main executable
#####################################################################################

add_executable(kinfu_viewer ${KINFU_VIEWER_SOURCES} ${KINFU_VIEWER_RES} ${KINFU_VIEWER_MOC_SRC} ${KINFU_VIEWER_UI_HDRS})
target_link_libraries(kinfu_viewer ${KINFU_VIEWER_DEPENDENCIES} )


