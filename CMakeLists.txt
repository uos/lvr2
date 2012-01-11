cmake_minimum_required(VERSION 2.8)
project(LSSR)

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )

link_directories("${CMAKE_SOURCE_DIR}/lib")
link_directories("$ENV{HOME}/local/lib")

include_directories(${LSSR_SOURCE_DIR}/ext/Eigen)
include_directories(${LSSR_SOURCE_DIR}/ext/stann)
include_directories(${LSSR_SOURCE_DIR})
include_directories(${LSSR_SOURCE_DIR}/src/lib)

set( CMAKE_CXX_FLAGS         "-O3 -g" )
set( CMAKE_CXX_FLAGS_RELEASE "-O3 -msse3 -Wno-deprecated-declarations -Wno-unused -Wcpp" )
set( CMAKE_CXX_FLAGS_DEBUG   "-g -Wall" )

###############################################################
# EXTERNAL LIBRARIES
###############################################################

find_package(PCL 1.0)
if(PCL_FOUND)
  message(STATUS "Found PCL")
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})
  add_definitions(-D_USE_PCL_)
endif(PCL_FOUND)

find_package( Boost 
    COMPONENTS 
    program_options 
    filesystem system 
    thread 
    REQUIRED )
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


# Try to find an installed rply library.
# If nothing is found we will build it ourself.
find_library( rply_shared
    NAMES rply
    PATHS /usr/lib /usr/local/lib /lib )
if( rply_shared )
    message(STATUS "Looking for rply - found")
else( rply_shared )
    message(STATUS "Looking for rply - not found")
    message(STATUS "Building local rply")
    include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/ext/rply/ )
    add_subdirectory(ext/rply)
endif( rply_shared )

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

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}" )
set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${OpenMP_CXX_FLAGS}" )

##############################################################
# USED THIRD PARTY STUFF
##############################################################

# RPLY
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/ext/rply/ )
add_subdirectory(ext/rply)

# LASLIB

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/ext/laslib/inc )
add_subdirectory(ext/laslib)

##############################################################
# LIBRARIES
##############################################################

add_subdirectory(src/lib)

##############################################################
# BINARIES
##############################################################

add_subdirectory(src/reconstruct)
add_subdirectory(src/colorize)
add_subdirectory(src/scanreduce)
add_subdirectory(src/convert)

if(PCL_FOUND)
    add_subdirectory(src/scanfilter)
endif(PCL_FOUND)

##############################################################
# VIEWERS
##############################################################

set( QT_USE_QTOPENGL TRUE )
set( QT_USE_XML TRUE)
find_package( Qt4 )
if( QT4_FOUND )
    include_directories(${QT_INCLUDE_DIR})
    include_directories(${QT_QTGUI_INCLUDE_DIR})
    include_directories(${QT_QTOPENGL_INCLUDE_DIR})
    include_directories(${QT_QTCORE_INCLUDE_DIR})
    include_directories(${QT_QTXML_INCLUDE_DIR})

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

# Try to find doxygen
find_package( Doxygen )

# Doxygen found
if( NOT DOXYGEN_FOUND )
    message( WARNING "Could not find Doxygen. "
        "You will not be able to build the documentation." )
endif( NOT DOXYGEN_FOUND )

# GraphViz found
if( NOT DOXYGEN_DOT_FOUND )
    message( WARNING "Could not find GraphViz. "
        "You will not be able to build the documentation." )
endif( NOT DOXYGEN_DOT_FOUND )

if( DOXYGEN_FOUND AND DOXYGEN_DOT_FOUND )
    configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in 
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY )
    add_custom_target( doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM )
endif( DOXYGEN_FOUND AND DOXYGEN_DOT_FOUND )
