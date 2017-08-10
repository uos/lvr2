cmake_minimum_required(VERSION 2.8)
project(LAS_VEGAS)

set_property(GLOBAL PROPERTY USE_FOLDERS On)

include(GNUInstallDirs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_MODULE_PATH
  ${LAS_VEGAS_SOURCE_DIR}/CMakeModules
  ${LAS_VEGAS_SOURCE_DIR}/ext/kintinuous/cmake/Modules
  ${CMAKE_MODULE_PATH}
)

message(STATUS ${CMAKE_MODULE_PATH})

include_directories(include)

##################################################################################
# Some compiler specific settings
##################################################################################
if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0")
    set(CMAKE_C_FLAGS_DEBUG "-O0")
endif(CMAKE_COMPILER_IS_GNUCC)

if(MSVC)
  list(APPEND LVR_DEFINITIONS -DNOMINMAX)
else(MSVC)
  add_compile_options(-fmessage-length=0 -fPIC -Wno-deprecated)
  # with cmake 3.3
  #add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++0x>)
  set(CMAKE_CXX_STANDARD 14)
endif(MSVC)

###############################################################
# EXTERNAL LIBRARIES
###############################################################

####
## Searching for CUDA
##############################
find_package( CUDA )


####
## Searching for OpenCV
##############################
find_package( OpenCV 2.4.8 REQUIRED )

option(WITH_CV_NONFREE "Use OpenCV non-free descriptors" OFF)
if(WITH_CV_NONFREE)
  message(STATUS "Using OpenCV non-free module")
  list(APPEND LVR_DEFINITIONS -DLVR_USE_CV_NONFREE)
endif(WITH_CV_NONFREE)


####
## Searching for FLANN
##############################
find_package(  FLANN REQUIRED )
if(FLANN_FOUND)
  include_directories(${FLANN_DIR})
endif(FLANN_FOUND)

find_package(  Lz4 REQUIRED )
include_directories(  ${LZ4_INCLUDE_DIR} )
message("LZ4 library found: ${LZ4_LIBRARY}")


####
## Searching for GSL
##############################
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})

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
set(VTK_QT_VERSION 5)
find_package(VTK REQUIRED)
if(VTK_MAJOR_VERSION VERSION_LESS "6")
  list(APPEND LVR_DEFINITIONS -DLVR_USE_VTK5)
endif()

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

#find_package(PCL 1.7)
#if(PCL_FOUND)
#  include_directories(${PCL_INCLUDE_DIRS})
#  link_directories(${PCL_LIBRARY_DIRS})
#  add_definitions(${PCL_DEFINITIONS})
#  list(APPEND LVR_DEFINITIONS -DLVR_USE_PCL)
#else(PCL_FOUND)
#  message( "-- No PCL found." )
#  message( "-- PCL related stuff will be disabled." )
#endif(PCL_FOUND)

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

if(MPI_FOUND)
find_package( Boost 1.54
    COMPONENTS
    program_options
    filesystem
    system
    thread
    serialization
    mpi
    REQUIRED )
else()

find_package( Boost 1.54
    COMPONENTS
    program_options
    filesystem
    system
    thread
    serialization
    REQUIRED )
endif()

link_directories(${Boost_LIBRARY_DIR})
include_directories(${Boost_INCLUDE_DIRS})
add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})

####
## Searching for OpenMP
##############################

find_package(OpenMP)
if(OPENMP_FOUND)
  link_directories(${OpenMP_LIBRARY_DIRS})
  include_directories(${OpenMP_INCLUDE_DIRS})
  list(APPEND LVR_DEFINITIONS -DLVR_USE_OPEN_MP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

####
## Searching for OpenGL
##############################

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

if(APPLE)
  include_directories(/System/Library/Frameworks/GLUT.framework/Headers)
  include_directories(/System/Library/Frameworks/OpenGL.framework/Headers)
  IF(EXISTS "/opt/local/lib")
    link_directories(/opt/local/lib)
  endif()
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
option(WITH_FREENECT "Compile with libfreenect grabber" OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBFREENECT libfreenect)
if(WITH_FREENECT AND LIBFREENECT_FOUND)
  include_directories(${LIBFREENECT_INCLUDE_DIRS})
endif()

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

# GSL
include_directories("ext/gsl/include")

# SLAM6D stuff
add_subdirectory(ext/slam6d)
include_directories( ${SLAM6D_INCLUDE_DIR} )
list(APPEND LVR_INSTALL_INCLUDE_DIRS ${SLAM6D_INSTALL_INCLUDE_DIR})

##############################################################
# ADD LVR DEFINITIONS
##############################################################

add_definitions(${LVR_DEFINITIONS})

##############################################################
# LVR-Kinfu Checks
##############################################################
option(WITH_KINFU "Compile LVR Kinfu" OFF)

# Check and set CUDA host compiler flags. CUDA 6.5 is only
# compatible to gcc4.8. Older CUDA versions require GCC lower
# than 4.8

if(WITH_KINFU AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CUDA_VERSION VERSION_LESS 7)
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8 )
        message(STATUS "******************************************************************")
        message(STATUS "* Your gcc version needs to be lower than 4.9 to compile the LVR *")
        message(STATUS "* KinectFusion library and apps with CUDA 6.5 or less. Please    *")
        message(STATUS "* install gcc 4.8 and set the CUDA host compiler accordingly     *")
        message(STATUS "* by setting the CUDA_HOST_COMPILER to gcc4.8. This can be done  *")
        message(STATUS "* in your cmake gui program or by manually calling e.g.          *")
        message(STATUS "* 'cmake <SOURCE_DIR> -DCUDA_HOST_COMPILER=/usr/bin/gcc-4.8'     *")
        message(STATUS "******************************************************************")
        message(WARNING "Incompatible CUDA host compiler detected. Fix it manually.")
    endif()
endif()

if(CUDA_FOUND AND "${OpenCV_VERSION_PATCH}" VERSION_GREATER "8" AND WITH_KINFU)
    message(STATUS "Building LVR KinFu.")
    add_subdirectory(ext/kintinuous)
    include_directories(${LAS_VEGAS_SOURCE_DIR}/ext/kintinuous/kfusion/include)
endif()

##############################################################
# APPLE OMP OPTION FOR CLANG-OMP
##############################################################

if(APPLE AND (NOT OPENMP_FOUND))
        message(STATUS "******************************************************************")
        message(STATUS "* It seems you are trying to comile on OSX with an compiler that *")
        message(STATUS "* does not support OpenMP. To get maximum performance consider   *")
        message(STATUS "* installing clang-omp from homebrew (brew install clang-omp) and*")
        message(STATUS "* configure with 'cmake -DWITH_CLANG_OMP ..'                     *")
        message(STATUS "******************************************************************")

  option(WITH_CLANG_OMP "Compile with clang-omp")

  if(WITH_CLANG_OMP)
    set(CMAKE_C_COMPILER /usr/local/bin/clang-omp CACHE STRING "C compiler" FORCE)
      set(CMAKE_CXX_COMPILER /usr/local/bin/clang-omp++ CACHE STRING "C++ compiler" FORCE)
      list(APPEND LVR_DEFINITIONS -DLVR_USE_OPEN_MP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  endif()

endif()

##############################################################
# HEADERS
##############################################################

install(DIRECTORY include/lvr DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/lvr2 DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
list(APPEND LVR_INSTALL_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})


##############################################################
# VTK use file include. Has to be called after all CUDA code
# was generated and compiled. The VTK definitions corrupt the
# nvcc call
##############################################################

include(${VTK_USE_FILE})

##############################################################
# VIEWERS
##############################################################

find_package(Qt5 COMPONENTS Core Widgets Xml OpenGL)
if(Qt5_FOUND)
  # Try to find libQGLViewer
  find_library(QGLVIEWER_LIBRARY_RELEASE
      NAMES qglviewer-qt4 qglviewer QGLViewer QGLViewer2 libQGLViewer-qt4
      PATHS /usr/lib /usr/local/lib ENV QGLVIEWERROOT ENV LD_LIBRARY_PATH ENV LIBRARY_PATH
      PATH_SUFFIXES QGLViewer QGLViewer/release)
  if(QGLVIEWER_LIBRARY_RELEASE)
    message(STATUS "Found libqglviewer")
    message(STATUS "Building qglviewer application")
    add_subdirectory(src/qglviewer)
  endif(QGLVIEWER_LIBRARY_RELEASE)

  find_package(QVTK)
  if(QVTK_FOUND)
    include_directories(${QVTK_INCLUDE_DIR})
    add_subdirectory(src/tools/viewer)
  endif(QVTK_FOUND)
endif(Qt5_FOUND)

##############################################################
# LIBRARIES
##############################################################

add_subdirectory(src/liblvr)

##############################################################
# BINARIES
##############################################################

add_subdirectory(src/tools/reconstruct)
add_subdirectory(src/tools/reconstruct2)
add_subdirectory(src/tools/classifier)
add_subdirectory(src/tools/colorize)
add_subdirectory(src/tools/scanreduce)
add_subdirectory(src/tools/convert)
add_subdirectory(src/tools/asciiconverter)
add_subdirectory(src/tools/texman)
add_subdirectory(src/tools/tiogen)
add_subdirectory(src/tools/meshoptimizer)
add_subdirectory(src/tools/transform)
add_subdirectory(src/tools/registration)
add_subdirectory(src/tools/normals)
add_subdirectory(src/tools/kaboom)
add_subdirectory(src/tools/image_normals)

if(MPI_FOUND)
  add_subdirectory(src/tools/mpi)
endif(MPI_FOUND)

if(CGAL_FOUND)
  # There is a name clash between CGAL and std::pair. Should
  # be fixed in the next CGAL release
  if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_COMPILER_IS_GNU)
    add_subdirectory(src/tools/cgal_reconstruction)
  endif()
  #add_subdirectory(src/tools/fusion)
  add_subdirectory(src/tools/slicer)
endif(CGAL_FOUND)

if(PCL_FOUND)
  add_subdirectory(src/tools/scanfilter)
  if(LIBFREENECT_FOUND)
    add_subdirectory(src/tools/kinectgrabber)
  endif(LIBFREENECT_FOUND)
endif(PCL_FOUND)

################################################################
# CMAKE FILES
################################################################

install(EXPORT lvrTargets
    FILE lvrTargets.cmake
    NAMESPACE lvr::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lvr)

configure_file(LVRConfig.cmake.in ${CMAKE_BINARY_DIR}/LVRConfig.cmake @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/LVRConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lvr)

install(FILES
    CMakeModules/FindEigen3.cmake
    CMakeModules/FindNabo.cmake
    CMakeModules/FindSTANN.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lvr/Modules)

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
