cmake_minimum_required(VERSION 3.4)
project(LAS_VEGAS VERSION 2)

set_property(GLOBAL PROPERTY USE_FOLDERS On)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

# Add -O0 to remove optimizations when using gcc
if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
    #add_compile_options(-O0 -g)
endif(CMAKE_COMPILER_IS_GNUCC)

if(MSVC)
  list(APPEND LVR2_DEFINITIONS -DNOMINMAX)
else(MSVC)
  add_compile_options(-fmessage-length=0 -fPIC -Wno-deprecated)
  # with cmake 3.3
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++11>)
  set(CMAKE_CXX_STANDARD 14)
endif(MSVC)

###############################################################
# EXTERNAL LIBRARIES
###############################################################

###
## Searching for TIFF
##############################
find_package(TIFF REQUIRED)
if(TIFF_FOUND)
    include_directories(${TIFF_INCLUDE_DIRS})
endif(TIFF_FOUND)

###
## Searching for GDAL
##############################
find_package(GDAL REQUIRED)
if(GDAL_FOUND)
    include_directories(${GDAL_INCLUDE_DIR})
endif(GDAL_FOUND)

####
## Searching for CUDA
##############################
if(NOT APPLE)
find_package( CUDA 8)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)
endif(NOT APPLE)

####
## Searching for OpenCV
##############################
find_package( OpenCV 3 )
if (!OpenCV_FOUND)
  find_package( OpenCV )
endif()
include_directories( ${OpenCV_INCLUDE_DIRS} )

option(WITH_CV_NONFREE "Use OpenCV non-free descriptors" OFF)
if(WITH_CV_NONFREE)
  message(STATUS "Using OpenCV non-free module")
  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_CV_NONFREE)
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

###
## Searching for OpenCl
##############################
find_package(OpenCL)
if(OPENCL_FOUND)
    include_directories(OPENCL_INCLUDE_DIR)
endif()

####
## Searching for VTK
##############################
set(VTK_QT_VERSION 5)
find_package(VTK REQUIRED)


if(DEFINED VTK_MAJOR_VERSION AND VTK_MAJOR_VERSION VERSION_LESS "6")
  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_VTK5)
endif()

if(VTK_MAJOR_VERSION VERSION_EQUAL "7")
    if(VTK_MINOR_VERSION VERSION_GREATER "0")
        list(APPEND LVR2_DEFINITIONS -DLVR2_USE_VTK_GE_7_1)
    endif()
endif()

if(VTK_MAJOR_VERSION VERSION_GREATER "7")
  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_VTK_GE_7_1)
endif()

if(VTK_VERSION VERSION_GREATER "7")
  
  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_VTK_GE_7_1)
  set(VTK_COMPONENTS "${VTK_AVAILABLE_COMPONENTS}")
  set(VTK_HAS_QT OFF)
  foreach(VTK_COMPONENT ${VTK_COMPONENTS})
    if("${VTK_COMPONENT}" STREQUAL "GUISupportQt")
      set(VTK_HAS_QT ON)
    endif()
  endforeach()

  message(STATUS "VTK > 7 found")
  if(VTK_HAS_QT)
    message(STATUS "VTK Component QVTK found")
  endif()
endif()

####
## Searching for NABO
##############################

find_package(Nabo)
if(NABO_FOUND)
  include_directories(${NABO_INCLUDE_DIRS})
  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_NABO)
endif(NABO_FOUND)

####
## Searching for PCL
##############################

#find_package(PCL 1.7)
#if(PCL_FOUND)
#  include_directories(${PCL_INCLUDE_DIRS})
#  link_directories(${PCL_LIBRARY_DIRS})
#  add_definitions(${PCL_DEFINITIONS})
#  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_PCL)
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
    mpi
    serialization
    timer
    iostreams
    REQUIRED )
else()
find_package( Boost 1.54
    COMPONENTS
    program_options
    filesystem
    system
    thread
    mpi
    serialization
    timer
    iostreams
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
  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_OPEN_MP)
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
  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_STANN)
  include_directories(${STANN_INCLUDE_DIR})
endif(STANN_FOUND)


##############################################################
# USED THIRD PARTY STUFF
##############################################################

# NanoFLANN
add_subdirectory(ext/nanoflann)
include_directories( ${NANOFLANN_INCLUDE_DIR} )
list(APPEND LVR2_INSTALL_INCLUDE_DIRS ${NANOFLANN_INSTALL_INCLUDE_DIR})

# PSIMPL
add_subdirectory(ext/psimpl)
include_directories( ${PSIMPL_INCLUDE_DIR} )
list(APPEND LVR2_INSTALL_INCLUDE_DIRS ${PSIMPL_INSTALL_INCLUDE_DIR})

# RPLY
add_subdirectory(ext/rply)
include_directories( ${RPLY_INCLUDE_DIR} )
list(APPEND LVR2_INSTALL_INCLUDE_DIRS ${RPLY_INSTALL_INCLUDE_DIR})

# LASLIB
add_subdirectory(ext/laslib)
include_directories( ${LASLIB_INCLUDE_DIR} )
list(APPEND LVR2_INSTALL_INCLUDE_DIRS ${LASLIB_INSTALL_INCLUDE_DIR})

# GSL
include_directories("ext/gsl/include")

# draco
add_subdirectory(ext/draco)
include_directories(ext/draco/src/)
include_directories(${CMAKE_BINARY_DIR})

# HighFive
set(HIGHFIVE_EXAMPLES FALSE CACHE INTERNAL "")
set(HIGHFIVE_UNIT_TESTS FALSE CACHE INTERNAL "")
add_subdirectory(ext/HighFive)
include_directories("ext/HighFive/include")
find_package(HDF5 REQUIRED COMPONENTS C CXX HL)
include_directories(${HDF5_INCLUDE_DIRS})
link_directories(${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

# SLAM6D stuff
add_subdirectory(ext/slam6d)
include_directories( ${SLAM6D_INCLUDE_DIR} )
list(APPEND LVR2_INSTALL_INCLUDE_DIRS ${SLAM6D_INSTALL_INCLUDE_DIR})

# CTPL
add_subdirectory(ext/CTPL)
include_directories(${CTPL_INCLUDE_DIR})
list(APPEND LVR2_INSTALL_INCLUDE_DIRS ${CTPL_INSTALL_INCLUDE_DIR})

# RiVLib
set(RiVLib_DIR ext/rivlib/cmake)
if(EXISTS ${PROJECT_SOURCE_DIR}/${RiVLib_DIR})
    set(RiVLib_USE_STATIC_RUNTIME ON)
    find_package(RiVLib COMPONENTS scanifc)
    if(RiVLib_FOUND)
        include_directories(${RiVLib_INCLUDE_DIRS})
        list(APPEND LVR2_DEFINITIONS -DLVR2_USE_RIVLIB)
        message(STATUS "Found RiVLib")
    endif()
endif()

##############################################################
# ADD LVR DEFINITIONS
##############################################################

add_definitions(${LVR2_DEFINITIONS})

##############################################################
# LVR-Kinfu Checks
##############################################################
option(WITH_KINFU "Compile LVR Kinfu" OFF)

# Check and set CUDA host compiler flags. CUDA 6.5 is only
# compatible to gcc4.8. Older CUDA versions require GCC lower
# than 4.8

if(CUDA_FOUND AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CUDA_VERSION VERSION_GREATER 8)
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7 )
        message(STATUS "******************************************************************")
        message(STATUS "* Your gcc version needs to be lower than 6 to compile the LVR   *")
        message(STATUS "* CUDA library and apps with CUDA. Setting CUDA_HOST_COMPILER to *")
        message(STATUS "* g++-6*. Please ensure that g++ 6 is installed on your system.  *")
        message(STATUS "******************************************************************")
        set(CUDA_HOST_COMPILER "g++-6" CACHE STRING "" FORCE)
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
      list(APPEND LVR2_DEFINITIONS -DLVR2_USE_OPEN_MP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  endif()

endif()

##############################################################
# HEADERS
##############################################################

install(DIRECTORY include/lvr2 DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
list(APPEND LVR2_INSTALL_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})


##############################################################
# VTK use file include. Has to be called after all CUDA code
# was generated and compiled. The VTK definitions corrupt the
# nvcc call
##############################################################

if(DEFINED VTK_USE_FILE)
  include(${VTK_USE_FILE})
endif()

##############################################################
# VIEWERS
##############################################################

find_package(Qt5 COMPONENTS Core Widgets Xml OpenGL)
if(Qt5_FOUND)
  find_package(QVTK)
  if(QVTK_FOUND)
    include_directories(${QVTK_INCLUDE_DIR})
    add_subdirectory(src/tools/lvr2_viewer)
  elseif(DEFINED VTK_VERSION AND VTK_VERSION VERSION_GREATER "7" AND VTK_HAS_QT)
    add_subdirectory(src/tools/lvr2_viewer)
  endif(QVTK_FOUND)
endif(Qt5_FOUND)

##############################################################
# LIBRARIES
##############################################################

add_subdirectory(src/liblvr2)

##############################################################
# BINARIES
##############################################################

add_subdirectory(src/tools/lvr2_reconstruct)
add_subdirectory(src/tools/lvr2_asciiconverter)
add_subdirectory(src/tools/lvr2_transform)
# doesnt compile
add_subdirectory(src/tools/lvr2_kaboom)
add_subdirectory(src/tools/lvr2_octree_test)
add_subdirectory(src/tools/lvr2_image_normals)
add_subdirectory(src/tools/lvr2_plymerger)
add_subdirectory(src/tools/lvr2_hdf5_builder)
add_subdirectory(src/tools/lvr2_hdf5_mesh_builder)
add_subdirectory(src/tools/lvr2_slam2hdf5)
add_subdirectory(src/tools/lvr2_hdf5togeotiff)
add_subdirectory(src/tools/lvr2_slam6d_merger)
add_subdirectory(src/tools/lvr2_scanproject2hdf5)

add_subdirectory(src/tools/lvr2_channel_test)

if (RiVLib_FOUND)
    add_subdirectory(src/tools/lvr2_riegl_project_converter)
endif()

if(CUDA_FOUND)
#    cuda_include_directories(ext/CTPL)
    add_subdirectory(src/tools/lvr2_cuda_normals)
endif()

if(OPENCL_FOUND)
    add_subdirectory(src/tools/lvr2_cl_normals)
endif()

################################################################
# CMAKE FILES
################################################################

install(EXPORT lvr2Targets
    FILE lvr2Targets.cmake
    NAMESPACE lvr2::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lvr2)

include(CMakePackageConfigHelpers)
set(INCLUDE_INSTALL_DIR include/)
set(LIB_INSTALL_DIR lib/)
set(SYSCONFIG_INSTALL_DIR etc/lvr2/)

configure_package_config_file(LVR2Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/LVR2Config.cmake
  INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/lvr2
  PATH_VARS INCLUDE_INSTALL_DIR SYSCONFIG_INSTALL_DIR)
  
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/LVR2ConfigVersion.cmake
  COMPATIBILITY SameMajorVersion
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/LVR2Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/LVR2ConfigVersion.cmake
              DESTINATION ${LIB_INSTALL_DIR}/cmake/lvr2 )

install(FILES
    CMakeModules/FindEigen3.cmake
    CMakeModules/FindFLANN.cmake
    CMakeModules/FindLZ4.cmake
    CMakeModules/FindNabo.cmake
    CMakeModules/FindOpenNI.cmake
    CMakeModules/FindOpenNI2.cmake
    CMakeModules/FindQVTK.cmake
    CMakeModules/FindSTANN.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lvr2/Modules)

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
