cmake_minimum_required(VERSION 2.8)
project(LSSR)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )

link_directories("${CMAKE_SOURCE_DIR}/lib")
link_directories("$ENV{HOME}/local/lib")

include_directories(${LSSR_SOURCE_DIR}/ext/Eigen)
include_directories(${LSSR_SOURCE_DIR}/ext/stann)
include_directories(${LSSR_SOURCE_DIR})
include_directories(${LSSR_SOURCE_DIR}/src/lib)

set(ADDITIONAL_COMPILER_FLAGS "-g -O4 -msse3 -fpermissive -Wno-deprecated -Wno-write-strings -lrply")

###############################################################
# EXTERNAL LIBRARIES
###############################################################

find_package(PCL 1.0 REQUIRED)
if(PCL_FOUND)
  message(STATUS "Found PCL")
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})
endif(PCL_FOUND)

find_package( Boost COMPONENTS program_options filesystem system thread REQUIRED)
if(Boost_FOUND)
  link_directories(${BOOST_LIBRARY_DIRS})
  include_directories(${Boost_INCLUDE_DIRS})
  add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})
endif()

find_package(OpenMP)
if(OpenMP_FOUND)
  link_directories(${OpenMP_LIBRARY_DIRS})
  include_directories(${OpenMP_INCLUDE_DIRS})
endif(OpenMP_FOUND)

find_package(OpenGL REQUIRED)
if(OpenGL_FOUND)
  link_directories(${OpenGL_INCLUDE_DIRS})
  include_directories(${OpenGL_LIBRARY_DIRS})
endif(OpenGL_FOUND)

IF(APPLE)
  link_directories(/opt/local/lib)
  include_directories(/System/Library/Frameworks/GLUT.framework/Headers)
  include_directories(/System/Library/Frameworks/OpenGL.framework/Headers)
ENDIF(APPLE)

if(UNIX)
  FIND_PATH(OPENGL_INC gl.h /usr/include/GL)
  include_directories(${OPENGL_INC})
endif(UNIX)

find_package(GLUT REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ADDITIONAL_COMPILER_FLAGS} ${OpenMP_CXX_FLAGS}")

##############################################################
# LIBRARIES
##############################################################

add_subdirectory(src/lib/model3d)
add_subdirectory(src/lib)

##############################################################
# BINARIES
##############################################################

add_subdirectory(src/reconstruct)
add_subdirectory(src/scanreduce)

##############################################################
# VIEWERS
##############################################################

set(QT_USE_QTOPENGL TRUE)
find_package(Qt4)
if(QT4_FOUND) 
  include_directories(${QT_INCLUDE_DIR})
  include_directories(${QT_QTGUI_INCLUDE_DIR})
  include_directories(${QT_QTOPENGL_INCLUDE_DIR})
  include_directories(${QT_QTCORE_INCLUDE_DIR})
  include_directories(${QT_QTXML_INCLUDE_DIR})
  add_subdirectory(src/viewer)

  # Try to find libQGLViewer
  find_library(QGLVIEWER_LIBRARY_RELEASE 
		NAMES qglviewer-qt4 qglviewer QGLViewer QGLViewer2 
		PATHS /usr/lib /usr/local/lib ENV QGLVIEWERROOT 
		ENV LD_LIBRARY_PATH ENV LIBRARY_PATH
		PATH_SUFFIXES QGLViewer QGLViewer/release)
  if(QGLVIEWER_LIBRARY_RELEASE)
    message(STATUS "Found libqglviewer")
    message(STATUS "Building qglviewer application")
    add_subdirectory(src/qglviewer)
  endif(QGLVIEWER_LIBRARY_RELEASE)
endif(QT4_FOUND)

################################################################
# DOCUMENTATION
################################################################

#Try to find doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc
${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)
endif(DOXYGEN_FOUND)





