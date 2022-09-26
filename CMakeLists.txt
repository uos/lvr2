cmake_minimum_required(VERSION 3.10)
project(LAS_VEGAS VERSION 2)

if(MSVC)
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
        CACHE STRING "Vcpkg toolchain file")
endif()

# OPTIONS
option(BUILD_EXAMPLES "Build the examples" OFF)

set_property(GLOBAL PROPERTY USE_FOLDERS On)
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()

#------------------------------------------------------------------------------
# Searching for Embree: needs to be placed above "set(CMAKE_MODULE_PATH)".
#------------------------------------------------------------------------------
find_package(embree 3.0 QUIET)
if(embree_FOUND)
  message(STATUS "Found Embree")
  include_directories(${EMBREE_INCLUDE_DIRS})
  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_EMBREE)
endif(embree_FOUND)

set(CMAKE_MODULE_PATH
  ${LAS_VEGAS_SOURCE_DIR}/CMakeModules
  ${LAS_VEGAS_SOURCE_DIR}/ext/kintinuous/cmake/Modules
  ${CMAKE_MODULE_PATH}
)

message(STATUS ${CMAKE_MODULE_PATH})

include_directories(include)

###############################################################################
# Compiler specific settings
###############################################################################

# Add -O0 to remove optimizations when using gcc
if(CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
endif(CMAKE_COMPILER_IS_GNUCC)

if(MSVC)
  list(APPEND LVR2_DEFINITIONS -DNOMINMAX)
    set(CMAKE_CXX_STANDARD 17)
    set(HAVE_CXX_ATOMICS64_WITHOUT_LIB True)
    set(HAVE_CXX_ATOMICS_WITHOUT_LIB True)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  add_compile_definitions(_HAS_STD_BYTE=0)
  add_compile_definitions(__TBB_NO_IMPLICIT_LINKAGE)
  #add_compile_definitions(__TBBMALLOC_NO_IMPLICIT_LINKAGE=1)
  add_compile_definitions(_USE_MATH_DEFINES)
else(MSVC)
  add_compile_options(-fmessage-length=0 -fPIC -Wno-deprecated)
  # with cmake 3.3
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++1z>)
  set(CMAKE_CXX_STANDARD 17)
endif(MSVC)

###############################################################################
# EXTERNAL LIBRARIES
###############################################################################

#####################################################################################
# RDB
#####################################################################################


# das hier muss verändert werden damit es läuft WICHTIG !!!!!
#set (rdb_ROOT_DIR "/home/praktikum/Desktop/rdb_test/rdblib-2.4.0-x86_64-linux"
#        CACHE PATH "path to RDB ROOT directory ")
#set(rdb_DIR "${rdb_ROOT_DIR}/interface/cpp")

find_package(RDB REQUIRED)

#set(RDB_LIB_DIR "${rdb_ROOT_DIR}/library")
#set(RDB_INCLUDE_C_DIR "${rdb_ROOT_DIR}/interface/c")
#set(RDB_INCLUDE_CPP_DIR "${rdb_ROOT_DIR}/interface/cpp")
if(RDB_FOUND)
    include_directories(${RDB_INCLUDE_DIRS})
    message(STATUS "Found LibRDB at ${RDB_INCLUDE_DIRS}, ${RDB_LIBRARIES}.")

    # TODO: Should we really do this?
    install(TARGETS rdbcpp
            EXPORT lvr2Targets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            )
endif()


#------------------------------------------------------------------------------
# Searching for Thread Building Blocks library
#------------------------------------------------------------------------------
find_package(TBB REQUIRED)
if(TBB_FOUND)
    message(STATUS "Found TBB library")
endif(TBB_FOUND)
#------------------------------------------------------------------------------
# Searching for TIFF
#------------------------------------------------------------------------------
find_package(TIFF REQUIRED)
if(TIFF_FOUND)
    message(STATUS "Found TIFF library")
    include_directories(${TIFF_INCLUDE_DIRS})
endif(TIFF_FOUND)

#------------------#####################################################################################
# RDB
#####################################################################################



find_package(GDAL REQUIRED)
if(GDAL_FOUND)
message(STATUS "Found GDAL")
    include_directories(${GDAL_INCLUDE_DIR})
endif(GDAL_FOUND)

#------------------------------------------------------------------------------
# Searching for CUDA
#------------------------------------------------------------------------------
if(MSVC)
    option(WITH_CUDA "Compile with CUDA support" FALSE)
else()
    option(WITH_CUDA "Compile with CUDA support" TRUE)
endif()
if(NOT APPLE AND WITH_CUDA)
find_package( CUDA 8)
if(CUDA_FOUND)
  message(STATUS "Found CUDA")
    include_directories(${CUDA_INCLUDE_DIRS})
    list(APPEND LVR2_DEFINITIONS -DLVR2_USE_CUDA)
endif(CUDA_FOUND)
endif(NOT APPLE AND WITH_CUDA)

#------------------------------------------------------------------------------
# Searching for OpenCV
#------------------------------------------------------------------------------
find_package( OpenCV 3 QUIET )
if (NOT OpenCV_FOUND)
  find_package( OpenCV 4)
  if(OpenCV_FOUND)
	message(STATUS "Found OpenCV 4")
  endif()
else()
  message(STATUS "Found OpenCV 3")
endif()
include_directories( ${OpenCV_INCLUDE_DIRS} )

option(WITH_CV_NONFREE "Use OpenCV non-free descriptors" OFF)
if(WITH_CV_NONFREE)
  message(STATUS "Using OpenCV non-free module")
  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_CV_NONFREE)
endif(WITH_CV_NONFREE)


#------------------------------------------------------------------------------
# Searching for FLANN
#------------------------------------------------------------------------------
find_package(FLANN REQUIRED)
if(FLANN_FOUND)
	message(STATUS "Found FLANN library: ${FLANN_INCLUDE_DIR}")
	include_directories(${FLANN_INCLUDRE_DIR})
endif(FLANN_FOUND)

find_package(Lz4)
if(LZ4_FOUND)
  include_directories(  ${LZ4_INCLUDE_DIR} )
  message(STATUS "Found LZ4 library: ${LZ4_INCLUDE_DIR}")
else(LZ4_FOUND)
  find_package(PkgConfig)
  pkg_search_module(LZ4 lz4 liblz4)
  include_directories(${LZ4_INCLUDEDIR})
  message(STATUS "Found LZ4 library: ${LZ4_INCLUDE_DIR}")
endif(LZ4_FOUND)


#------------------------------------------------------------------------------
# Searching for GSL
#------------------------------------------------------------------------------
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})
message(STATUS "Found GSL")

#------------------------------------------------------------------------------
# Searching for Eigen3
#------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "Found Eigen3")

#------------------------------------------------------------------------------
# Searching for OpenCl
#------------------------------------------------------------------------------
find_package(OpenCL2)
if(OPENCL_FOUND)
  message(STATUS "Found OpenCL")
    include_directories(OPENCL_INCLUDE_DIR)
    list(APPEND LVR2_DEFINITIONS -DLVR2_USE_OPENCL)
    if(OpenCL_NEW_API)
      list(APPEND LVR2_DEFINITIONS -DLVR2_USE_OPENCL_NEW_API)
    endif(OpenCL_NEW_API)
endif(OPENCL_FOUND)

#------------------------------------------------------------------------------
# Searching for VTK
#------------------------------------------------------------------------------
set(VTK_QT_VERSION 5)
find_package(VTK REQUIRED)

if(DEFINED VTK_MAJOR_VERSION AND VTK_MAJOR_VERSION VERSION_EQUAL "8")
  # patched QVTKOpenGL header because of name clash with signals.
  add_subdirectory(ext/QVTKOpenGLWidget)
  include_directories(BEFORE ${QVTK_PATCHED_INCLUDE_DIR} )
  list(APPEND LVR2_INSTALL_INCLUDE_DIRS ${QVTK_PATCHED_INCLUDE_DIR})
  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_VTK8)
elseif(DEFINED VTK_MAJOR_VERSION AND VTK_MAJOR_VERSION VERSION_EQUAL "9")
  add_subdirectory(ext/QVTKOpenGLWidget)
  include_directories(BEFORE ${QVTK_PATCHED_INCLUDE_DIR} )
  list(APPEND LVR2_INSTALL_INCLUDE_DIRS ${QVTK_PATCHED_INCLUDE_DIR})
  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_VTK9)
endif()
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

  message(STATUS "Found VTK (Version > 7)")
  if(VTK_HAS_QT)
    message(STATUS "Found VTK compoment QVTK")
  endif()
endif()

#------------------------------------------------------------------------------
# Searching for NABO
#------------------------------------------------------------------------------
find_package(Nabo)
if(Nabo_FOUND)
  include_directories(${Nabo_INCLUDE_DIRS})
  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_NABO)
endif(Nabo_FOUND)

#------------------------------------------------------------------------------
## Searching for PCL
#------------------------------------------------------------------------------
option(WITH_PCL "Compile with PCL support" OFF)
if(WITH_PCL)
    find_package(PCL)
    if(PCL_FOUND)
        # PCL only results in linker error due to internal mpi dependency
        find_package(MPI)
        if(MPI_FOUND)
            message(STATUS "PCL and MPI found. Compile with PCL support.")
            include_directories(${PCL_INCLUDE_DIRS})
            link_directories(${PCL_LIBRARY_DIRS})
            add_definitions(${PCL_DEFINITIONS})
            list(APPEND LVR2_DEFINITIONS -DLVR2_USE_PCL)
        else(MPI_FOUND)
            message(WARNING "PCL found but no mpi. To use PCL support please install MPI")
            unset(PCL_FOUND)
        endif(MPI_FOUND)
    else(PCL_FOUND)
        message(STATUS "-- No PCL found." )
        message(STATUS "-- PCL related stuff will be disabled." )
    endif(PCL_FOUND)
endif(WITH_PCL)

find_package(MPI)
if(MPI_FOUND)
    message(STATUS "Found MPI")
    include_directories(${MPI_CXX_INCLUDE_PATH})
endif(MPI_FOUND)

#------------------------------------------------------------------------------
# Searching CGAL
#------------------------------------------------------------------------------

find_package(CGAL)
if(CGAL_FOUND)
  set(_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS)
  # CGAL_USE_FILE overrides theses flags...
  include(${CGAL_USE_FILE})
  set(CMAKE_CXX_FLAGS "${_CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

  message(STATUS "Found CGAL")
endif(CGAL_FOUND)

#------------------------------------------------------------------------------
# Searching for Boost
#------------------------------------------------------------------------------

#set(Boost_USE_STATIC_LIBS       ON)
set(Boost_USE_MULTITHREADED     ON)
set(Boost_COMPONENTS
  program_options
  filesystem
  system
  thread
  serialization
  timer
  iostreams
  date_time)

if(MPI_FOUND)
  set($Boost_COMPONENTS ${Boost_COMPONENTS} mpi)
  message(STATUS "Found MPI. Adding Boost MPI library to required components.")
endif()

find_package(Boost COMPONENTS ${Boost_COMPONENTS} REQUIRED QUIET)
message(STATUS "Found Boost libraries")

link_directories(${Boost_LIBRARY_DIR})
include_directories(${Boost_INCLUDE_DIRS})
add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})

#------------------------------------------------------------------------------
# Searching for OpenMP
#------------------------------------------------------------------------------

find_package(OpenMP)
if(OPENMP_FOUND)
  message(STATUS "Found OpenMP")
  link_directories(${OpenMP_LIBRARY_DIRS})
  include_directories(${OpenMP_INCLUDE_DIRS})
  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_OPEN_MP)

  # tasks only in OpenMP 3.0 and above thus we need llvm openmp in experimental mode
  # See here: https://devblogs.microsoft.com/cppblog/improved-openmp-support-for-cpp-in-visual-studio/
  if(MSVC)
    set(CMAKE_CXX_FLAGS " -openmp:llvm -openmp:experimental ${CMAKE_CXX_FLAGS}")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

#------------------------------------------------------------------------------
# Searching for OpenGL
#------------------------------------------------------------------------------

find_package(OpenGL COMPONENTS OpenGL)
include_directories(${OPENGL_INCLUDE_DIR})
message(STATUS "Found OpenGL: ${OPENGL_INCLUDE_DIR}")

if(APPLE)
  include_directories(/System/Library/Frameworks/GLUT.framework/Headers)
  include_directories(/System/Library/Frameworks/OpenGL.framework/Headers)
  IF(EXISTS "/opt/local/lib")
    link_directories(/opt/local/lib)
  endif()
endif(APPLE)

#------------------------------------------------------------------------------
## Searching for glut
#------------------------------------------------------------------------------

find_package(GLUT)
if(GLUT_FOUND)
  message(STATUS "Found OpenGL Utility Toolkit via cmake: ${GLUT_INCLUDE_DIRS}" )
  include_directories(${GLUT_INCLUDE_DIRS})
  
else(GLUT_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GLUT glut freeglut FREEGLUT REQUIRED)
  message(STATUS "Found OpenGL Utility Toolkit via pkg_config: ${GLUT_LIBDIR} ${FREEGLUT_LIBDIR}" )
  include_directories(${GLUT_INCLUDEDIR})
  link_directories(${GLUT_LIBDIR})
  set(GLUT_LIBRARIES ${pkgcfg_lib_GLUT_glut})
  message(STATUS "Using ugly cmake cache hack to provide GLUT libraries: ": ${GLUT_LIBRARIES})
endif(GLUT_FOUND)

#------------------------------------------------------------------------------
## Searching for libfreenect
#------------------------------------------------------------------------------
option(WITH_FREENECT "Compile with libfreenect grabber" OFF)

find_package(PkgConfig REQUIRED)
if(WITH_FREENECT )
  pkg_check_modules(LIBFREENECT libfreenect)
  if(LIBFREENECT_FOUND)
    message(STATUS "Found Freenect")
    include_directories(${LIBFREENECT_INCLUDE_DIRS})
  endif()
endif()

#------------------------------------------------------------------------------
## Searching for STANN
#------------------------------------------------------------------------------
set(STANN_DIR "" CACHE PATH "directory that contains the STANN library headers")
find_package(STANN)
if(STANN_FOUND)
  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_STANN)
  include_directories(${STANN_INCLUDE_DIR})
  message(STATUS "Found STANN")
endif(STANN_FOUND)

###############################################################################
# USED THIRD PARTY STUFF
###############################################################################

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

# HDF5
find_package(HDF5 REQUIRED COMPONENTS C CXX HL)
include_directories(${HDF5_INCLUDE_DIRS})
message(STATUS "Found HDF5")

# HighFive
set(HIGHFIVE_INCLUDE_DIRS "ext/HighFive/include")
list(APPEND LVR2_INSTALL_INCLUDE_DIRS ${HIGHFIVE_INCLUDE_DIRS})
include_directories(${HIGHFIVE_INCLUDE_DIRS})

# yaml-cpp
find_package(yaml-cpp REQUIRED)
# no found variable
include_directories(${YAML_CPP_INCLUDE_DIR})

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

# Cesium-native
option(WITH_3DTILES "Compile with 3DTiles support" OFF)
if(WITH_3DTILES)
  include(ExternalProject)
  ExternalProject_Add(cesium-native
    PREFIX cesium-native
    GIT_REPOSITORY "https://github.com/CesiumGS/cesium-native"
    GIT_TAG "v0.16.0"
    LOG_CONFIGURE ON
    LOG_INSTALL ON # Hide cmake output (creates a lot of spam on every build)
    LOG_BUILD ON
    LOG_OUTPUT_ON_FAILURE ON
    CMAKE_ARGS
        "-DCMAKE_INSTALL_PREFIX=install"  # install into the build directory rather than /usr/local/... (which would require sudo)
                                          # this is in quotes because otherwise cmake adds a random "x86_64-linux-gnu" to lib paths. No idea why.
        -DCESIUM_TESTS_ENABLED=OFF        # don't build the tests
        -DDRACO_TRANSCODER_SUPPORTED=ON   # enable Gltf support for draco
        -DCMAKE_CXX_FLAGS=-fPIC           # tell cmake to compile as shared libraries
  )
  ExternalProject_Get_Property(cesium-native BINARY_DIR)
  include_directories(${BINARY_DIR}/install/include)
  link_directories(${BINARY_DIR}/install/lib)

  ExternalProject_Get_Property(cesium-native SOURCE_DIR)
  include_directories(${SOURCE_DIR}/extern/draco/src
                      ${SOURCE_DIR}/extern/draco/third_party/tinygltf
                      ${BINARY_DIR}    # draco generates include files in BINARY_DIR
  )
  set(draco_FOUND ON)
  set(draco_LIBRARIES draco)

  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_3DTILES -DLVR2_USE_DRACO)

  set(3DTILES_LIBRARIES Cesium3DTiles Cesium3DTilesWriter CesiumGltf CesiumGltfWriter CesiumJsonWriter)

else(WITH_3DTILES)
  #------------------------------------------------------------------------------
  # Searching for Draco
  #------------------------------------------------------------------------------
  find_package(Draco)
  if(draco_FOUND)
    message(STATUS "Found Draco")
    include_directories(${draco_INCLUDE_DIRS})
    list(APPEND LVR2_DEFINITIONS -DLVR2_USE_DRACO)
  endif(draco_FOUND)
endif(WITH_3DTILES)

###############################################################################
# ADD LVR DEFINITIONS
###############################################################################

add_definitions(${LVR2_DEFINITIONS})

###############################################################################
# LVR-Kinfu Checks
###############################################################################
option(WITH_KINFU "Compile LVR Kinfu" OFF)


# ###############################################################################
# # Check and set CUDA host compiler flags.          
# ###############################################################################
if(CUDA_FOUND)

include(max_cuda_gcc_version)
max_cuda_gcc_version(CUDA_VERSION MAX_CUDA_GCC_VERSION)
message(STATUS "Highest supported GCC version for CUDA: ${MAX_CUDA_GCC_VERSION}")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER ${MAX_CUDA_GCC_VERSION} )
    message(STATUS "******************************************************************")
    message(STATUS "* Your gcc version needs to be lower than or equal ${MAX_CUDA_GCC_VERSION} to compile  *")
    message(STATUS "* the CUDA library and apps. Setting CUDA_HOST_COMPILER to       *")
    message(STATUS "* g++-${MAX_CUDA_GCC_VERSION}. Please ensure that g++-${MAX_CUDA_GCC_VERSION} is installed on your system.   *")
    message(STATUS "******************************************************************")
    set(CUDA_HOST_COMPILER "g++-${MAX_CUDA_GCC_VERSION}" CACHE STRING "" FORCE)
  endif()
endif()


###############################################################################
# APPLE OMP OPTION FOR CLANG-OMP
###############################################################################
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


###############################################################################
# HEADERS
###############################################################################

install(DIRECTORY include/lvr2 DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
list(APPEND LVR2_INSTALL_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})



###############################################################################
# VTK use file include. Has to be called after all CUDA code
# was generated and compiled. The VTK definitions corrupt the
# nvcc call
###############################################################################

if(DEFINED VTK_USE_FILE)
  include(${VTK_USE_FILE})
endif()


###############################################################################
# LVR2 VIEWER CHECKS
###############################################################################

find_package(Qt5 COMPONENTS Core Widgets Xml OpenGL)
if(Qt5_FOUND)
   message(STATUS "Found QT5")
   find_package(QVTK)
   if(QVTK_FOUND)
     message(STATUS "Found QVTK ${QVTK_INCLUDE_DIR}")
     include_directories(${QVTK_INCLUDE_DIR})
     add_subdirectory(src/tools/lvr2_viewer)
   elseif(DEFINED VTK_VERSION AND VTK_VERSION VERSION_GREATER "7" AND VTK_HAS_QT)
     add_subdirectory(src/tools/lvr2_viewer)
   endif(QVTK_FOUND)
endif(Qt5_FOUND)

###############################################################################
# LVR2 ASCII VIEWER CHECKS
###############################################################################
find_package(Curses)
if(CURSES_FOUND)
    include_directories( ${CURSES_INCLUDE_DIRS} )
endif()

###############################################################################
# SET DEPENDENCIES
###############################################################################

if(MSVC)
set(LVR2_LIB_DEPENDENCIES
    ${Boost_LIBARIES}
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_THREAD_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_IOSTREAMS_LIBRARY}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${GDAL_LIBRARY}
    ${OpenCV_LIBS}
    ${GSL_LIBRARIES}
    ${LZ4_LIBRARY}
    ${TIFF_LIBRARY}
    ${YAML_CPP_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES}
    lvr2rply
    lvr2rply_static
    lvr2las
    lvr2las_static 
    )
else()
set(LVR2_LIB_DEPENDENCIES
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_THREAD_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_IOSTREAMS_LIBRARY}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${GDAL_LIBRARY}
    ${OpenCV_LIBS}
    ${GSL_LIBRARIES}
    ${LZ4_LIBRARY}
    ${TIFF_LIBRARY}
    ${YAML_CPP_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
    ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES}
    m
    ${RDB_LIBRARIES}
    )
endif()

#####################################################################################
# ADD ALL EXTERNAL DEPENDENCIES
#####################################################################################

if(UNIX)
list(APPEND LVR2_LIB_DEPENDENCIES ${LVR2_LIB_DEPENDENCIES} pthread)
endif(UNIX)

if(PCL_FOUND)
list(APPEND LVR2_LIB_DEPENDENCIES ${PCL_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif(PCL_FOUND)

if(WITH_FREENECT AND LIBFREENECT_FOUND)
list(APPEND LVR2_LIB_DEPENDENCIES ${LIBFREENECT_LIBRARIES})
endif()

if(draco_FOUND)
list(APPEND LVR2_LIB_DEPENDENCIES ${draco_LIBRARIES})
endif(draco_FOUND)

if(RiVLib_FOUND)
list(APPEND LVR2_LIB_DEPENDENCIES ${RiVLib_SCANIFC_LIBRARY})
endif()

if(OPENCL_FOUND)
list(APPEND LVR2_LIB_DEPENDENCIES ${OpenCL_LIBRARIES})
endif()

if(MPI_FOUND)
  set(LVR2_LIB_DEPENDENCIES ${LVR2_LIB_DEPENDENCIES} ${MPI_LIBRARIES})
endif()

if(MSVC)
    list(APPEND LVR2_LIB_DEPENDENCIES yaml-cpp)
endif()
if(YAML_CPP_LIBRARIES)
list(APPEND LVR2_LIB_DEPENDENCIES ${YAML_CPP_LIBRARIES})
endif()

if(embree_FOUND)
list(APPEND LVR2_LIB_DEPENDENCIES ${EMBREE_LIBRARY})
endif()

if(WITH_3DTILES)
    list(APPEND LVR2_LIB_DEPENDENCIES ${3DTILES_LIBRARIES})
endif(WITH_3DTILES)

###############################################################################
# LIBRARIES
###############################################################################

add_subdirectory(src/liblvr2)

###############################################################################
# BINARIES
###############################################################################

if(NOT MSVC)
add_subdirectory(src/tools/lvr2_largescale_reconstruct)
endif(NOT MSVC)

add_subdirectory(src/tools/lvr2_reconstruct)
add_subdirectory(src/tools/lvr2_dmc_reconstruction)
add_subdirectory(src/tools/lvr2_gs_reconstruction)
add_subdirectory(src/tools/lvr2_asciiconverter)
add_subdirectory(src/tools/lvr2_transform)
add_subdirectory(src/tools/lvr2_kaboom)
add_subdirectory(src/tools/lvr2_octree_test)
add_subdirectory(src/tools/lvr2_image_normals)
add_subdirectory(src/tools/lvr2_plymerger)
add_subdirectory(src/tools/lvr2_hdf5_mesh_builder)
#add_subdirectory(src/tools/lvr2_slam2hdf5)
#add_subdirectory(src/tools/lvr2_hdf5togeotiff)
add_subdirectory(src/tools/lvr2_slam6d_merger)
add_subdirectory(src/tools/lvr2_chunking)
#add_subdirectory(src/tools/lvr2_registration)
add_subdirectory(src/tools/lvr2_mesh_reducer)
add_subdirectory(src/tools/lvr2_chunking_server)
add_subdirectory(src/tools/lvr2_scanproject_parser)
add_subdirectory(src/tools/lvr2_ground_level_extractor)
# add_subdirectory(src/tools/lvr2_hdf5_convert_old)
add_subdirectory(src/tools/lvr2_hdf5_inspect)
add_subdirectory(src/tools/lvr2_3dtiles)



if(CURSES_FOUND AND embree_FOUND)
    add_subdirectory(src/tools/lvr2_ascii_viewer)
endif()

if (RiVLib_FOUND)
    #add_subdirectory(src/tools/lvr2_riegl_project_converter)
endif()

if(CUDA_FOUND)
#    cuda_include_directories(ext/CTPL)
    add_subdirectory(src/tools/lvr2_cuda_normals)
endif()

if(OPENCL_FOUND)
    add_subdirectory(src/tools/lvr2_cl_normals)
    add_subdirectory(src/tools/lvr2_cl_sor)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()


###############################################################################
# CMAKE FILES
###############################################################################

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

# install package.xml for ROS
install(FILES package.xml DESTINATION share/lvr2)

install(FILES
    CMakeModules/FindEigen3.cmake
    CMakeModules/FindFLANN.cmake
    CMakeModules/FindLz4.cmake
    CMakeModules/FindNabo.cmake
    CMakeModules/FindOpenNI.cmake
    CMakeModules/FindOpenNI2.cmake
    CMakeModules/FindQVTK.cmake
    CMakeModules/FindSTANN.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lvr2/Modules)

# Installation of HighFive headers
# need to do it like this sind add_subdirectory to "ext/HighFive" crashs
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ext/HighFive/include/
    DESTINATION "include"
    PATTERN "*.in" EXCLUDE)

###############################################################################
# DOCUMENTATION
###############################################################################

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
