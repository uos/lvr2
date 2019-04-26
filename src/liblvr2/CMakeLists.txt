set(LVR2_SOURCES
    config/lvropenmp.cpp
    config/BaseOption.cpp
    texture/Texture.cpp
    texture/TextureFactory.cpp
    util/Util.cpp
    display/PointOctree.cpp
    display/Renderable.cpp
    display/GroundPlane.cpp
    display/MultiPointCloud.cpp
    display/StaticMesh.cpp
    display/Color.cpp
    display/Grid.cpp
    display/PointCloud.cpp
    display/InteractivePointCloud.cpp
    display/CoordinateAxes.cpp
    display/ColorMap.cpp
    display/GlTexture.cpp
    display/PointCorrespondences.cpp
    display/Arrow.cpp
    display/TexturedMesh.cpp
    display/MeshCluster.cpp
    io/AsciiIO.cpp
    io/CoordinateTransform.cpp
    io/ObjIO.cpp
#    io/KinectIO.cpp
    io/AttributeMeshIOBase.cpp
    io/PPMIO.cpp
    io/PLYIO.cpp
    io/IOUtils.cpp
    io/ChannelManager.cpp
    io/STLIO.cpp
    io/UosIO.cpp
    io/PCDIO.cpp
    io/Progress.cpp
    io/MeshBuffer.cpp
    io/DracoEncoder.cpp
    io/DracoDecoder.cpp
    io/DrcIO.cpp
#    io/KinectGrabber.cpp
    io/DatIO.cpp
    io/LasIO.cpp
    io/BaseIO.cpp
    io/HDF5IO.cpp
    io/Timestamp.cpp
    io/BoctreeIO.cpp
    io/GridIO.cpp
    io/BaseBuffer.cpp
    io/PointBuffer.cpp
    io/ModelFactory.cpp
    io/ScanprojectIO.cpp
    io/ScanData.cpp
    io/ScanDataManager.cpp
    reconstruction/Projection.cpp
    reconstruction/PanoramaNormals.cpp
    reconstruction/ModelToImage.cpp
    reconstruction/opencl/ClSurface.cpp
    reconstruction/LBKdTree.cpp
    reconstruction/cuda/CudaSurface.cu
    reconstruction/PCLFiltering.cpp
    registration/EigenSVDPointAlign.cpp
    registration/ICPPointAlign.cpp
)



#####################################################################################
# Create source groups to structure code in Visual Studio
#####################################################################################

if(MSVC)
  include(LVRVSSetup.txt)
endif()

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_INTERNAL_DEPENDENCIES_STATIC
    lvr2rply_static
    lvr2las_static
    lvr2slam6d_static)

set(LVR2_INTERNAL_DEPENDENCIES_SHARED
    lvr2rply
    lvr2las
    lvr2slam6d)

set(LVR2_LIB_DEPENDENCIES
    draco
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_THREAD_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${OpenCV_LIBS}
    ${GSL_LIBRARIES}
    ${LZ4_LIBRARY})


#####################################################################################
# On Unix/Linux we need to link against pthreads
#####################################################################################

if(UNIX)
  set(LVR2_LIB_DEPENDENCIES ${LVR2_LIB_DEPENDENCIES} pthread)
endif(UNIX)

#####################################################################################
# Set c++0x flags for gcc compilers (needed for boctree io)
#####################################################################################

if(UNIX)
  SET_SOURCE_FILES_PROPERTIES(io/BoctreeIO.cpp PROPERTIES COMPILE_FLAGS "-std=c++14")
endif(UNIX)

#####################################################################################
# Add PCD io if PCL is installed
#####################################################################################

if(PCL_FOUND)
  set(LVR2_SOURCES ${LVR2_SOURCES}
      io/PCDIO.cpp
      reconstruction/PCLFiltering.cpp)

  set(LVR2_LIB_DEPENDENCIES ${LVR2_LIB_DEPENDENCIES} ${PCL_LIBRARIES})
endif(PCL_FOUND)

find_package(HDF5 REQUIRED C CXX HL)
include_directories(${HDF5_INCLUDE_DIR})

if(WITH_FREENECT AND LIBFREENECT_FOUND)
  set(LVR2_SOURCES ${LVR2_SOURCES} io/KinectGrabber.cpp io/KinectIO.cpp ../../include/lvr2/algorithm/UtilAlgorithms.hpp)
  set(LVR2_LIB_DEPENDENCIES ${LVR2_LIB_DEPENDENCIES} ${LIBFREENECT_LIBRARIES})
endif()

#####################################################################################
# RiVLib
#####################################################################################

if(RiVLib_FOUND)
    set(LVR2_SOURCES ${LVR2_SOURCES} io/RxpIO.cpp)
    set(LVR2_LIB_DEPENDENCIES ${LVR2_LIB_DEPENDENCIES} ${RiVLib_SCANIFC_LIBRARY})
endif()

#####################################################################################
# OpenCL
#####################################################################################

if(OPENCL_FOUND)
    set(LVR2_SOURCES ${LVR2_SOURCES} reconstruction/opencl/ClSurface.cpp)
    set(LVR2_LIB_DEPENDENCIES ${LVR2_LIB_DEPENDENCIES} ${OpenCL_LIBRARIES})
endif()

#####################################################################################
# CUDA
#####################################################################################

if(CUDA_FOUND)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

    # List of CUDA kernel code sources
    set(LVR2_CUDA_SRC
        reconstruction/cuda/CudaSurface.cu
    )

    # Fix broken VTK flags
    get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
    set(vtk_flags)
    foreach(it ${dir_defs})
        if(it MATCHES "vtk*")
        list(APPEND vtk_flags ${it})
        endif()
    endforeach()

    foreach(d ${vtk_flags})
        remove_definitions(-D${d})
    endforeach()

    message(STATUS "Building static LVR CUDA library")
    cuda_add_library(lvr2cuda_static STATIC ${LVR2_CUDA_SRC})
    target_link_libraries(lvr2cuda_static lvr2_static)

    # Add dependency to avoid that both targets
    # are build concurrently in parallel builds
    add_dependencies(lvr2cuda_static lvr2_static)
    message(STATUS "Building shared LVR CUDA library")
    cuda_add_library(lvr2cuda SHARED ${LVR2_CUDA_CPP_SRC} ${LVR2_CUDA_SRC})
    target_link_libraries(lvr2cuda lvr2)

    install(
        TARGETS lvr2cuda_static lvr2cuda
        EXPORT lvr2Targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

endif()

#####################################################################################
# Compile object files for static and dynamic library
#####################################################################################

add_library(lvr2core OBJECT ${LVR2_SOURCES})

#####################################################################################
# Build static library
#####################################################################################

message(STATUS "Building static library")
add_library(lvr2_static STATIC $<TARGET_OBJECTS:lvr2core>)
target_link_libraries(lvr2_static ${LVR2_INTERNAL_DEPENDENCIES_STATIC} ${LVR2_LIB_DEPENDENCIES} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})


#####################################################################################
# Build shared library
#####################################################################################

if( NOT MSVC )
  message(STATUS "Building shared library")
  add_library(lvr2 SHARED $<TARGET_OBJECTS:lvr2core>)
  target_link_libraries(lvr2 ${LVR2_INTERNAL_DEPENDENCIES_SHARED} ${LVR2_LIB_DEPENDENCIES} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

install(TARGETS lvr2_static lvr2
  EXPORT lvrTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif( NOT MSVC)
