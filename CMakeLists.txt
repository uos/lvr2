cmake_minimum_required(VERSION 2.8)
project(LAS_VEGAS)
SET(ENV{CC} "icc")
SET(ENV{CXX} "icpc")

set_property(GLOBAL PROPERTY USE_FOLDERS On)

include(GNUInstallDirs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_MODULE_PATH ${LAS_VEGAS_SOURCE_DIR}/CMakeModules ${CMAKE_MODULE_PATH})

##################################################################################
# Some compiler specific settings
##################################################################################

if(MSVC)
  list(APPEND LVR_DEFINITIONS -DNOMINMAX)
endif()

if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_COMPILER_IS_GNUCXX)
  add_compile_options(-fmessage-length=0 -fPIC -Wno-deprecated)
  # with cmake 3.3
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++0x>)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

###############################################################
# EXTERNAL LIBRARIES
###############################################################

####
## Searching for OpenCV
##############################
find_package( OpenCV REQUIRED )

####
## Searching for Eigen3
##############################
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

####
## Searching for MPI
##############################
find_package( MPI )
if(MPI_FOUND)
  include_directories(SYSTEM ${MPI_INCLUDE_PATH})
endif(MPI_FOUND)

####
## Searching for VTK
##############################
#find_package(VTK REQUIRED)
#include(${VTK_USE_FILE})
#if(VTK_MAJOR_VERSION VERSION_LESS "6")
#  list(APPEND LVR_DEFINITIONS -DLVR_USE_VTK5)
#endif()

####
## Searching for NABO
##############################

find_package(Nabo)
if(NABO_FOUND)
  include_directories(${NABO_INCLUDE_DIRS})
  list(APPEND LVR_DEFINITIONS -DLVR_USE_NABO)
endif(NABO_FOUND)

####
## Searching for PCL
##############################

find_package(PCL 1.6)
if(PCL_FOUND)
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})
  list(APPEND LVR_DEFINITIONS -DLVR_USE_PCL)
else(PCL_FOUND)
  message( "-- No PCL found." )
  message( "-- PCL related stuff will be disabled." )
endif(PCL_FOUND)

####
## Searching CGAL
##############################

find_package(CGAL)
if(CGAL_FOUND)
  set(_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS)
  # CGAL_USE_FILE overrides theses flags...
  include(${CGAL_USE_FILE})
  set(CMAKE_CXX_FLAGS "${_CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

  message(STATUS "FOUND CGAL")
endif(CGAL_FOUND)

####
## Searching for Boost
##############################

#set(Boost_USE_STATIC_LIBS       ON)
set(Boost_USE_MULTITHREADED     ON)

find_package( Boost 1.54
    COMPONENTS
    program_options
    filesystem
    system
    thread
    REQUIRED )

link_directories(${Boost_LIBRARY_DIR})
include_directories(${Boost_INCLUDE_DIRS})
add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})

####
## Searching for OpenMP
##############################

find_package(OpenMP)
if(OpenMP_FOUND)
  link_directories(${OpenMP_LIBRARY_DIRS})
  include_directories(${OpenMP_INCLUDE_DIRS})
  list(APPEND LVR_DEFINITIONS -DLVR_USE_OPEN_MP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OpenMP_FOUND)

####
## Searching for OpenGL
##############################

find_package(OpenGL REQUIRED)
message( "OPEN GL LIBS : " ${OPENGL_LIBRARIES})
include_directories(${OPENGL_INCLUDE_DIR})

if(APPLE)
  include_directories(/System/Library/Frameworks/GLUT.framework/Headers)
  include_directories(/System/Library/Frameworks/OpenGL.framework/Headers)
  link_directories(/opt/local/lib)
endif(APPLE)

####
## Searching for glut
##############################

find_package(GLUT REQUIRED)
message(STATUS "-- Found OpenGL Utility Toolkit" )
include_directories(${GLUT_INCLUDE_DIR})

####
## Searching for libfreenect
##############################

#find_package(PkgConfig REQUIRED)
#pkg_check_modules(LIBFREENECT libfreenect)
#if(LIBFREENECT_FOUND)
#  include_directories(${LIBFREENECT_INCLUDE_DIRS})
#endif(LIBFREENECT_FOUND)

####
## Searching for STANN
##############################

set(STANN_DIR "" CACHE PATH "directory that contains the STANN library headers")
find_package(STANN)
if(STANN_FOUND)
  list(APPEND LVR_DEFINITIONS -DLVR_USE_STANN)
  include_directories(${STANN_INCLUDE_DIR})
endif(STANN_FOUND)

##############################################################
# USED THIRD PARTY STUFF
##############################################################

# NanoFLANN
add_subdirectory(ext/nanoflann)
include_directories( ${NANOFLANN_INCLUDE_DIR} )
list(APPEND LVR_INSTALL_INCLUDE_DIRS ${NANOFLANN_INSTALL_INCLUDE_DIR})

# PSIMPL
add_subdirectory(ext/psimpl)
include_directories( ${PSIMPL_INCLUDE_DIR} )
list(APPEND LVR_INSTALL_INCLUDE_DIRS ${PSIMPL_INSTALL_INCLUDE_DIR})

# RPLY
add_subdirectory(ext/rply)
include_directories( ${RPLY_INCLUDE_DIR} )
list(APPEND LVR_INSTALL_INCLUDE_DIRS ${RPLY_INSTALL_INCLUDE_DIR})

# LASLIB
add_subdirectory(ext/laslib)
include_directories( ${LASLIB_INCLUDE_DIR} )
list(APPEND LVR_INSTALL_INCLUDE_DIRS ${LASLIB_INSTALL_INCLUDE_DIR})

# SLAM6D stuff
add_subdirectory(ext/slam6d)
include_directories( ${SLAM6D_INCLUDE_DIR} )
list(APPEND LVR_INSTALL_INCLUDE_DIRS ${SLAM6D_INSTALL_INCLUDE_DIR})



##############################################################
# HEADERS
##############################################################

include_directories(include)

install(DIRECTORY include/lvr DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

list(APPEND LVR_INSTALL_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})

add_definitions(${LVR_DEFINITIONS})

##############################################################
# VIEWERS
##############################################################



##############################################################
# LIBRARIES
##############################################################

add_subdirectory(src/liblvr)

##############################################################
# BINARIES
##############################################################

add_subdirectory(src/tools/reconstruct)
add_subdirectory(src/tools/classifier)
#add_subdirectory(src/tools/colorize)
add_subdirectory(src/tools/scanreduce)
add_subdirectory(src/tools/convert)
add_subdirectory(src/tools/asciiconverter)
add_subdirectory(src/tools/texman)
add_subdirectory(src/tools/tiogen)
add_subdirectory(src/tools/meshoptimizer)
add_subdirectory(src/tools/transform)
#add_subdirectory(src/tools/registration)
#add_subdirectory(src/tools/normals)

if(MPI_FOUND)
  add_subdirectory(src/tools/mpi)
endif(MPI_FOUND)

if(CGAL_FOUND)
  add_subdirectory(src/tools/cgal_reconstruction)
  #add_subdirectory(src/tools/fusion)
  add_subdirectory(src/tools/slicer)
endif(CGAL_FOUND)

if(PCL_FOUND)
  add_subdirectory(src/tools/scanfilter)
  add_subdirectory(src/tools/leica_converter)
endif(PCL_FOUND)

################################################################
# CMAKE FILES
################################################################

install(EXPORT lvrTargets
    FILE lvrTargets.cmake
    NAMESPACE lvr::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lvr)

configure_file(lvrConfig.cmake.in ${CMAKE_BINARY_DIR}/lvrConfig.cmake @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/lvrConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lvr)

################################################################
# DOCUMENTATION
################################################################

find_package( Doxygen )
if( NOT DOXYGEN_FOUND )
  message( WARNING "Could not find Doxygen. "
      "You will not be able to build the documentation." )
endif( NOT DOXYGEN_FOUND )

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

