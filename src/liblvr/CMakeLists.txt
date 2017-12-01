set(LVR_SOURCES
    config/lvropenmp.cpp
    io/BaseIO.cpp
    io/ModelFactory.cpp
    io/PLYIO.cpp
    io/Hdf5IO.cpp
    io/AsciiIO.cpp
    io/UosIO.cpp
    io/ObjIO.cpp
    io/LasIO.cpp
    io/PPMIO.cpp
    io/Progress.cpp
    io/Timestamp.cpp
    io/MeshBuffer.cpp
    io/PointBuffer.cpp
    io/GridIO.cpp
    io/CoordinateTransform.cpp
    io/BoctreeIO.cpp
    io/STLIO.cpp
    io/TextureIO.cpp
    io/DatIO.cpp
    io/IOUtils.cpp
    config/BaseOption.cpp
    display/InteractivePointCloud.cpp
    display/CoordinateAxes.cpp
    display/Renderable.cpp
    display/PointCloud.cpp
    display/MultiPointCloud.cpp
    display/StaticMesh.cpp
    display/ColorMap.cpp
    display/GroundPlane.cpp
    display/Grid.cpp
    display/Color.cpp
    display/MeshCluster.cpp
    display/GlTexture.cpp
    display/TextureFactory.cpp
    display/TexturedMesh.cpp
    registration/EigenSVDPointAlign.cpp
    registration/ICPPointAlign.cpp
    reconstruction/ModelToImage.cpp
    reconstruction/Projection.cpp
    reconstruction/PanoramaNormals.cpp
    reconstruction/LBKdTree.cpp
    texture/Texture.cpp
    texture/ImageProcessor.cpp
    texture/Statistics.cpp
    texture/AutoCorr.cpp
    texture/CrossCorr.cpp
    texture/CCV.cpp
    texture/Transform.cpp
    texture/Trans.cpp
    geometry/HalfEdgeAccessExceptions.cpp
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

set(LVR_INTERNAL_DEPENDENCIES_STATIC
    lvrrply_static
    lvrlas_static
    lvrslam6d_static)

set(LVR_INTERNAL_DEPENDENCIES_SHARED
    lvrrply
    lvrlas
    lvrslam6d)

set(LVR_LIB_DEPENDENCIES
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
  set(LVR_LIB_DEPENDENCIES ${LVR_LIB_DEPENDENCIES} pthread)
endif(UNIX)

#####################################################################################
# Set c++0x flags for gcc compilers (needed for boctree io)
#####################################################################################

if(UNIX)
  SET_SOURCE_FILES_PROPERTIES(io/BoctreeIO.cpp PROPERTIES COMPILE_FLAGS "-std=gnu++0x")
endif(UNIX)

#####################################################################################
# Add PCD io if PCL is installed
#####################################################################################

if(PCL_FOUND)
  set(LVR_SOURCES ${LVR_SOURCES}
      io/PCDIO.cpp
      reconstruction/PCLFiltering.cpp)

  set(LVR_LIB_DEPENDENCIES ${LVR_LIB_DEPENDENCIES} ${PCL_LIBRARIES})
endif(PCL_FOUND)

if(WITH_FREENECT AND LIBFREENECT_FOUND)
  set(LVR_SOURCES ${LVR_SOURCES} io/KinectGrabber.cpp io/KinectIO.cpp ../../include/lvr2/algorithm/UtilAlgorithms.hpp)
  set(LVR_LIB_DEPENDENCIES ${LVR_LIB_DEPENDENCIES} ${LIBFREENECT_LIBRARIES})
endif()

#####################################################################################
# OpenCL
#####################################################################################

if(OPENCL_FOUND)
    set(LVR_SOURCES ${LVR_SOURCES} reconstruction/opencl/ClSurface.cpp)
    set(LVR_LIB_DEPENDENCIES ${LVR_LIB_DEPENDENCIES} ${OpenCL_LIBRARIES})
endif()

#####################################################################################
# CUDA
#####################################################################################

if(CUDA_FOUND)

    # List of CUDA kernel code sources
    set(LVR_CUDA_SRC
        reconstruction/cuda/CudaSurface.cu
    )

    # CUDA sources need to be compiled in C++11 mode
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

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
    cuda_add_library(lvrcuda_static STATIC ${LVR_CUDA_SRC})
    target_link_libraries(lvrcuda_static lvr_static)

    # Add dependency to avoid that both targets
    # are build concurrently in parallel builds
    add_dependencies(lvrcuda_static lvr_static)
    message(STATUS "Building shared LVR CUDA library")
    cuda_add_library(lvrcuda SHARED ${LVR_CUDA_CPP_SRC} ${LVR_CUDA_SRC})
    target_link_libraries(lvrcuda lvr)

    install(
        TARGETS lvrcuda_static lvrcuda
        EXPORT lvrTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

endif()

#####################################################################################
# Compile object files for static and dynamic library
#####################################################################################

add_library(lvrcore OBJECT ${LVR_SOURCES})

#####################################################################################
# Build static library
#####################################################################################

message(STATUS "Building static library")
add_library(lvr_static STATIC $<TARGET_OBJECTS:lvrcore>)
target_link_libraries(lvr_static ${LVR_INTERNAL_DEPENDENCIES_STATIC} ${LVR_LIB_DEPENDENCIES})


#####################################################################################
# Build shared library
#####################################################################################

if( NOT MSVC )
  message(STATUS "Building shared library")
  add_library(lvr SHARED $<TARGET_OBJECTS:lvrcore>)
  target_link_libraries(lvr ${LVR_INTERNAL_DEPENDENCIES_SHARED} ${LVR_LIB_DEPENDENCIES})

install(TARGETS lvr_static lvr
  EXPORT lvrTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif( NOT MSVC)
