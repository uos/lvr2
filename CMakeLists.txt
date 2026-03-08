cmake_minimum_required(VERSION 3.16)
project(lvr2 VERSION 25.2.2)

# OPTIONS
option(LVR2_BUILD_EXAMPLES "Build the examples" OFF)
option(LVR2_BUILD_VIEWER "Build lvr2_viewer" OFF)
option(LVR2_BUILD_TOOLS "Build tools including lvr2_reconstruct" ON)
option(LVR2_BUILD_TOOLS_EXPERIMENTAL "Build experimental tools" OFF)
option(LVR2_WITH_KINFU "Compile LVR Kinfu" OFF)
option(LVR2_WITH_3DTILES "Compile with 3DTiles support" OFF)
option(LVR2_WITH_CUDA "Compile with CUDA support, if available" ON)
option(LVR2_WITH_OPENCL "Compile with OpenCL support, if available" ON)
option(LVR2_WITH_PCL "Compile with PCL support" OFF)
option(LVR2_WITH_FREENECT "Compile with libfreenect grabber" OFF)
option(LVR2_WITH_CV_NONFREE "Use OpenCV non-free descriptors" OFF)

## Compile as C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# DEFAULT RELEASE
if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

if(MSVC)
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
        CACHE STRING "Vcpkg toolchain file")
endif()

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

#############################
# List of policies that where we need to force the old behavior
# -> This is also a list of TODOs. So ultimately this list should be empty
###

# exec_programm should not be called anymore (uninstall script)
if(POLICY CMP0153)
  cmake_policy(SET CMP0153 OLD)
endif()

# FindCUDA is obsolete
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

#------------------------------------------------------------------------------
# Searching for Embree: needs to be placed above "set(CMAKE_MODULE_PATH)".
#------------------------------------------------------------------------------
find_package(embree 4 QUIET)
if(embree_FOUND)
  message(STATUS "Found Embree")
  include_directories(${EMBREE_INCLUDE_DIRS})
  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_EMBREE)
  list(APPEND LVR2_DEFINITIONS -DLVR2_EMBREE_VERSION=4)
  set(LVR2_EMBREE_VERSION 4)
else(embree_FOUND)
  find_package(embree 3 QUIET)
  if(embree_FOUND)
    message(STATUS "Found Embree")
    include_directories(${EMBREE_INCLUDE_DIRS})
    list(APPEND LVR2_DEFINITIONS -DLVR2_USE_EMBREE)
    list(APPEND LVR2_DEFINITIONS -DLVR2_EMBREE_VERSION=3)
    set(LVR2_EMBREE_VERSION 3)
  endif(embree_FOUND)
endif(embree_FOUND)

set(CMAKE_MODULE_PATH
  ${PROJECT_SOURCE_DIR}/CMakeModules
  ${PROJECT_SOURCE_DIR}/ext/kintinuous/cmake/Modules
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

find_package(MPI)
if(MPI_FOUND)
    message(STATUS "Found MPI")
    include_directories(${MPI_CXX_INCLUDE_PATH})
endif(MPI_FOUND)

###############################################################################
# REQUIRED DEPENDENCIES
###############################################################################

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
include_directories(${TIFF_INCLUDE_DIRS})

#------------------------------------------------------------------------------
# Searching for GDAL
#------------------------------------------------------------------------------
find_package(GDAL REQUIRED)
include_directories(${GDAL_INCLUDE_DIR})

#------------------------------------------------------------------------------
# Searching for OpenCV
#------------------------------------------------------------------------------
set(OPEN_CV_COMPONENTS "core;imgproc;imgcodecs;features2d;calib3d")
find_package( OpenCV 3 QUIET COMPONENTS ${OPEN_CV_COMPONENTS} )
if (NOT OpenCV_FOUND)
  find_package( OpenCV 4 COMPONENTS ${OPEN_CV_COMPONENTS})
  if(OpenCV_FOUND)
	message(STATUS "Found OpenCV 4")
  endif()
else()
  message(STATUS "Found OpenCV 3")
endif()
include_directories( ${OpenCV_INCLUDE_DIRS} )

if(LVR2_WITH_CV_NONFREE)
  message(STATUS "Using OpenCV non-free module")
  list(APPEND LVR2_DEFINITIONS -DLVR2_USE_CV_NONFREE)
endif(LVR2_WITH_CV_NONFREE)
#------------------------------------------------------------------------------
# Searching for FLANN
#------------------------------------------------------------------------------
find_package(FLANN REQUIRED)
message(STATUS "Found FLANN library: ${FLANN_INCLUDE_DIR}")
include_directories(${FLANN_INCLUDE_DIR})

#------------------------------------------------------------------------------
# Searching for LZ4
#------------------------------------------------------------------------------
find_package(LZ4 REQUIRED)
include_directories(${LZ4_INCLUDE_DIR})
message(STATUS "Found LZ4 library: ${LZ4_INCLUDE_DIR}")

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
# include_directories(${EIGEN3_INCLUDE_DIR})
# message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")

#------------------------------------------------------------------------------
# Searching for Boost
#------------------------------------------------------------------------------
#set(Boost_USE_STATIC_LIBS       ON)
set(Boost_USE_MULTITHREADED     ON)
set(Boost_COMPONENTS
  program_options
  filesystem
  thread
  serialization
  timer
  iostreams
  date_time)

if(MPI_FOUND)
  set(Boost_COMPONENTS ${Boost_COMPONENTS} mpi)
  message(STATUS "Found MPI. Adding Boost MPI library to required components.")
endif()

find_package(Boost COMPONENTS ${Boost_COMPONENTS} CONFIG REQUIRED QUIET)
message(STATUS "Found Boost libraries")

link_directories(${Boost_LIBRARY_DIR})
include_directories(${Boost_INCLUDE_DIRS})
add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})

#------------------------------------------------------------------------------
# Searching for HDF5
#------------------------------------------------------------------------------
find_package(HDF5 REQUIRED COMPONENTS C CXX HL)
include_directories(${HDF5_INCLUDE_DIRS})
message(STATUS "Found HDF5")

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
  message(STATUS "Using ugly cmake cache hack to provide GLUT libraries: " : ${GLUT_LIBRARIES})
endif(GLUT_FOUND)

###############################################################################
# OPTIONAL DEPENDENCIES
###############################################################################

#------------------------------------------------------------------------------
# Searching for CUDA
#------------------------------------------------------------------------------
if(MSVC OR APPLE)
    set(LVR2_WITH_CUDA OFF CACHE BOOL "" FORCE)
endif(MSVC OR APPLE)

if(LVR2_WITH_CUDA)

  cmake_policy(SET CMP0104 NEW)

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()

    # default flags are not set when including cuda
  if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)

    if(NOT CUDA_NVCC_FLAGS_DEBUG)
      set(CUDA_NVCC_FLAGS_DEBUG "-g" CACHE STRING "" FORCE)
    endif()

    if(NOT CUDA_NVCC_FLAGS_MINSIZEREL)
      set(CUDA_NVCC_FLAGS_MINSIZEREL "-Os -DNDEBUG" CACHE STRING "" FORCE)
    endif()

    if(NOT CUDA_NVCC_FLAGS_RELEASE)
      set(CUDA_NVCC_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
    endif()

    if(NOT CUDA_NVCC_FLAGS_RELWITHDEBINFO)
      set(CUDA_NVCC_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG" CACHE STRING "" FORCE)
    endif()

  endif()

  # set(CUDA_STANDARD 14)

  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
      find_package(CUDAToolkit QUIET)
      if(CUDAToolkit_FOUND)
          enable_language(CUDA)
          set(CUDA_FOUND True)
          # for backwards compatibility
          set(CUDA_VERSION ${CUDAToolkit_VERSION})
          set(CUDA_INCLUDE_DIRS "") # is in target instead
          set(CUDA_LIBRARIES CUDA::cudart)
          set(CUDA_cusolver_LIBRARY CUDA::cusolver)
          set(CUDA_cublas_LIBRARY CUDA::cublas)
          set(CUDA_DRIVER_LIBRARY CUDA::cuda_driver)
          set(CUDA_CUDA_LIBRARY CUDA::cuda_driver)
          set(CUDA_NVRTC_LIBRARY CUDA::nvrtc)
      else()
          find_package(CUDA)
          if(CUDA_FOUND)
              enable_language(CUDA)
              set(CUDA_DRIVER_LIBRARY cuda)
          else()
              message(STATUS "Neither CudaToolkit nor CUDA found!")
          endif(CUDA_FOUND)
      endif(CUDAToolkit_FOUND)
  endif(CMAKE_CUDA_COMPILER)

  if(CUDA_FOUND)
    message(STATUS "Found CUDA")
    include_directories(${CUDA_INCLUDE_DIRS})
    list(APPEND LVR2_DEFINITIONS -DLVR2_USE_CUDA)
  endif(CUDA_FOUND)

endif(LVR2_WITH_CUDA)

#------------------------------------------------------------------------------
# Searching for RDB
#------------------------------------------------------------------------------
find_package(RDB)
if(RDB_FOUND)
    include_directories(${RDB_INCLUDE_DIRS})
    message(STATUS "Found LibRDB at ${RDB_INCLUDE_DIRS}, ${RDB_LIBRARIES}.")

    list(APPEND LVR2_DEFINITIONS -DLVR2_USE_RDB)
    
    install(TARGETS rdbcpp
            EXPORT lvr2-targets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            )
endif(RDB_FOUND)

#------------------------------------------------------------------------------
# Searching for OpenCl
#------------------------------------------------------------------------------
if(LVR2_WITH_OPENCL)
  find_package(OpenCL2)
  if(OPENCL_FOUND)
    message(STATUS "Found OpenCL")
      include_directories(${OPENCL_INCLUDE_DIR})
      list(APPEND LVR2_DEFINITIONS -DLVR2_USE_OPENCL)
      if(OpenCL_NEW_API)
        list(APPEND LVR2_DEFINITIONS -DLVR2_USE_OPENCL_NEW_API)
      endif(OpenCL_NEW_API)
  endif(OPENCL_FOUND)
endif(LVR2_WITH_OPENCL)

# TODO: I cannot move this section down to the 'add_subdirectory' part.
# There must be some side effects on the way there 
if(LVR2_BUILD_VIEWER)
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
endif(LVR2_BUILD_VIEWER)

#------------------------------------------------------------------------------
## Searching for PCL
#------------------------------------------------------------------------------
if(LVR2_WITH_PCL)
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
endif(LVR2_WITH_PCL)

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
## Searching for libfreenect
#------------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
if(LVR2_WITH_FREENECT)
  pkg_check_modules(LIBFREENECT libfreenect)
  if(LIBFREENECT_FOUND)
    message(STATUS "Found Freenect")
    include_directories(${LIBFREENECT_INCLUDE_DIRS})
  endif()
endif(LVR2_WITH_FREENECT)

###############################################################################
# USED THIRD PARTY STUFF
###############################################################################

# spdlog
add_subdirectory(ext/spdlog)

# spdmon
set(spdmon_BUILD_EXECUTABLE OFF CACHE BOOL "Build the spdmon example")
set(spdmon_BUILD_HEADERS_ONLY ON CACHE BOOL "Build spdmon header only lib")
set(spdmon_ENABLE_UNIT_TESTING OFF CACHE BOOL "Build spdmon unit tests")
set(spdmon_ENABLE_BENCHMARKING OFF CACHE BOOL "Build spdmon benchmark")
add_subdirectory(ext/spdmon)

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

# HighFive
set(HIGHFIVE_INCLUDE_DIRS "ext/HighFive/include")
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
    endif(RiVLib_FOUND)
endif()

# Cesium-native
if(LVR2_WITH_3DTILES)
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

else(LVR2_WITH_3DTILES)
  #------------------------------------------------------------------------------
  # Searching for Draco
  #------------------------------------------------------------------------------
  find_package(Draco)
  if(draco_FOUND)
    message(STATUS "Found Draco")
    include_directories(${draco_INCLUDE_DIRS})
    list(APPEND LVR2_DEFINITIONS -DLVR2_USE_DRACO)
  endif(draco_FOUND)
endif(LVR2_WITH_3DTILES)

###############################################################################
# ADD LVR DEFINITIONS
###############################################################################

add_definitions(${LVR2_DEFINITIONS})

###############################################################################
# LVR-Kinfu Checks
###############################################################################


# ###############################################################################
# # Check and set CUDA host compiler flags.          
# ###############################################################################
if(CUDA_FOUND)
  include(max_cuda_gcc_version)
  max_cuda_gcc_version(CUDA_VERSION MAX_CUDA_GCC_VERSION)
  message(STATUS "Highest supported GCC version for CUDA: ${MAX_CUDA_GCC_VERSION}")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER ${MAX_CUDA_GCC_VERSION})
      message(STATUS "******************************************************************")
      message(STATUS "* Your gcc version needs to be lower than or equal ${MAX_CUDA_GCC_VERSION} to compile  *")
      message(STATUS "* the CUDA library and apps. Setting CUDA_HOST_COMPILER to       *")
      message(STATUS "* g++-${MAX_CUDA_GCC_VERSION}. Please ensure that g++-${MAX_CUDA_GCC_VERSION} is installed on your system.   *")
      message(STATUS "******************************************************************")
      set(CUDA_HOST_COMPILER "g++-${MAX_CUDA_GCC_VERSION}" CACHE STRING "" FORCE)
    endif()
endif(CUDA_FOUND)


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

  option(LVR2_WITH_CLANG_OMP "Compile with clang-omp")

  if(LVR2_WITH_CLANG_OMP)
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
    Eigen3::Eigen
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
    Eigen3::Eigen
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

if(LIBFREENECT_FOUND)
  list(APPEND LVR2_LIB_DEPENDENCIES ${LIBFREENECT_LIBRARIES})
endif(LIBFREENECT_FOUND)

if(draco_FOUND)
  list(APPEND LVR2_LIB_DEPENDENCIES ${draco_LIBRARIES})
endif(draco_FOUND)

if(RiVLib_FOUND)
  list(APPEND LVR2_LIB_DEPENDENCIES ${RiVLib_SCANIFC_LIBRARY})
endif(RiVLib_FOUND)

if(OPENCL_FOUND)
  list(APPEND LVR2_LIB_DEPENDENCIES ${OpenCL_LIBRARIES})
endif(OPENCL_FOUND)

if(MPI_FOUND)
  set(LVR2_LIB_DEPENDENCIES ${LVR2_LIB_DEPENDENCIES} ${MPI_LIBRARIES})
endif(MPI_FOUND)

if(MSVC)
  list(APPEND LVR2_LIB_DEPENDENCIES yaml-cpp)
endif(MSVC)

if(YAML_CPP_LIBRARIES)
  list(APPEND LVR2_LIB_DEPENDENCIES ${YAML_CPP_LIBRARIES})
endif(YAML_CPP_LIBRARIES)

if(embree_FOUND)
  list(APPEND LVR2_LIB_DEPENDENCIES ${EMBREE_LIBRARY})
endif(embree_FOUND)

if(3DTILES_FOUND)
  list(APPEND LVR2_LIB_DEPENDENCIES ${3DTILES_LIBRARIES})
endif(3DTILES_FOUND)

###############################################################################
# LIBRARIES
###############################################################################

add_subdirectory(src/liblvr2)

###############################################################################
# BINARIES
###############################################################################

if(NOT MSVC)
# add_subdirectory(src/tools/lvr2_largescale_reconstruct)
endif(NOT MSVC)

if(LVR2_BUILD_TOOLS)
  add_subdirectory(src/tools/lvr2_reconstruct)
  add_subdirectory(src/tools/lvr2_mesh_reducer)
  add_subdirectory(src/tools/lvr2_hdf5_mesh_tool)
endif(LVR2_BUILD_TOOLS)

if(LVR2_BUILD_TOOLS_EXPERIMENTAL)

  add_subdirectory(src/tools/lvr2_dmc_reconstruction)
  add_subdirectory(src/tools/lvr2_gs_reconstruction)
  add_subdirectory(src/tools/lvr2_asciiconverter)
  add_subdirectory(src/tools/lvr2_transform)
  add_subdirectory(src/tools/lvr2_kaboom)
  add_subdirectory(src/tools/lvr2_octree_test)
  add_subdirectory(src/tools/lvr2_image_normals)
  add_subdirectory(src/tools/lvr2_plymerger)
  
  # add_subdirectory(src/tools/lvr2_slam2hdf5)
  # add_subdirectory(src/tools/lvr2_hdf5togeotiff)
  add_subdirectory(src/tools/lvr2_slam6d_merger)
  add_subdirectory(src/tools/lvr2_chunking)
  # add_subdirectory(src/tools/lvr2_registration)
  add_subdirectory(src/tools/lvr2_chunking_server)
  add_subdirectory(src/tools/lvr2_scanproject_parser)
  add_subdirectory(src/tools/lvr2_ground_level_extractor)
  # add_subdirectory(src/tools/lvr2_hdf5_convert_old)
  add_subdirectory(src/tools/lvr2_hdf5_inspect)
  add_subdirectory(src/tools/lvr2_3dtiles)
  #add_subdirectory(src/tools/teaser_example)

  if(RiVLib_FOUND)
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

endif(LVR2_BUILD_TOOLS_EXPERIMENTAL)

###############################################################################
# LVR2 EXAMPLES
###############################################################################
if(LVR2_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

###############################################################################
# LVR2 VIEWER + LVR2 ASCII VIEWER
###############################################################################
if(LVR2_BUILD_VIEWER)

  if(DEFINED VTK_USE_FILE)
    include(${VTK_USE_FILE})
  endif()

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

  find_package(Curses)
  if(CURSES_FOUND)
      include_directories( ${CURSES_INCLUDE_DIRS} )
  endif()

  if(CURSES_FOUND AND embree_FOUND)
    add_subdirectory(src/tools/lvr2_ascii_viewer)
  endif()
endif(LVR2_BUILD_VIEWER)

###############################################################################
# CMAKE FILES
###############################################################################
include(CMakePackageConfigHelpers)
set(SYSCONFIG_INSTALL_DIR etc/lvr2/)

install(EXPORT lvr2-targets
    FILE lvr2-targets.cmake
    NAMESPACE lvr2::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lvr2)


write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/lvr2-config-version.cmake
  VERSION ${lvr2_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(CMakeModules/lvr2-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/lvr2-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lvr2
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/lvr2-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/lvr2-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lvr2
)

# install package.xml for ROS
install(FILES package.xml DESTINATION share/lvr2)

install(FILES
    CMakeModules/FindFLANN.cmake
    CMakeModules/FindLZ4.cmake
    CMakeModules/FindOpenNI.cmake
    CMakeModules/FindOpenNI2.cmake
    CMakeModules/FindQVTK.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lvr2/Modules)

# Installation of HighFive headers
# need to do it like this sind add_subdirectory to "ext/HighFive" crashs
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ext/HighFive/include/
    DESTINATION include
    PATTERN "*.in" EXCLUDE)


###############################################################################
# Uninstall
###############################################################################
# https://gitlab.kitware.com/cmake/community/-/wikis/FAQ#can-i-do-make-uninstall-with-cmake
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/lvr2-uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/lvr2-uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/lvr2-uninstall.cmake
    COMMENT "Uninstall lvr2 libraries and all header files")
endif()

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

# CPack / packaging configuration
include(CMakeModules/lvr2-packaging.cmake)
