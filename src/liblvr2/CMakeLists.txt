set(LVR2_SOURCES
    config/lvropenmp.cpp
    config/BaseOption.cpp
    texture/Texture.cpp
    texture/TextureFactory.cpp
    util/Util.cpp
    util/Hdf5Util.cpp
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
    io/STLIO.cpp
    io/UosIO.cpp
    io/PCDIO.cpp
    io/Progress.cpp
    io/MeshBuffer.cpp
    io/LineReader.cpp
#    io/KinectGrabber.cpp
    io/DatIO.cpp
    io/LasIO.cpp
    io/BaseIO.cpp
    io/GeoTIFFIO.cpp
    io/HDF5IO.cpp
    io/Timestamp.cpp
    io/BoctreeIO.cpp
    io/GridIO.cpp
    io/PointBuffer.cpp
    io/ModelFactory.cpp
    io/ScanDataManager.cpp
    io/ScanDirectoryParser.cpp
    io/ScanIOUtils.cpp
    io/descriptions/ScanProjectSchemaSLAM.cpp
    io/descriptions/ScanProjectSchemaHyperlib.cpp
    io/descriptions/DirectoryKernel.cpp
    io/descriptions/DirectoryIO.cpp
    io/descriptions/MetaFormatFactory.cpp
    io/descriptions/ScanProjectSchema.cpp
    io/descriptions/HDF5Kernel.cpp
    io/descriptions/HDF5IO.cpp
    io/descriptions/HDF5MetaDescriptionV2.cpp
    reconstruction/Projection.cpp
    reconstruction/PanoramaNormals.cpp
    reconstruction/ModelToImage.cpp
    reconstruction/LBKdTree.cpp
    algorithm/ChunkBuilder.cpp
    algorithm/ChunkManager.cpp
    algorithm/ChunkHashGrid.cpp
    registration/ICPPointAlign.cpp
    registration/KDTree.cpp
    registration/SLAMScanWrapper.cpp
    registration/Metascan.cpp
    registration/SLAMAlign.cpp
    registration/GraphSLAM.cpp
    registration/TreeUtils.cpp
    registration/OctreeReduction.cpp
    registration/RegistrationPipeline.cpp
)

#####################################################################################
# Add PCD io if PCL is installed
#####################################################################################

if(PCL_FOUND)
set(LVR2_SOURCES ${LVR2_SOURCES}
    io/PCDIO.cpp
    reconstruction/PCLFiltering.cpp)
endif(PCL_FOUND)

if(embree_FOUND)
list(APPEND LVR2_SOURCES 
    algorithm/raycasting/EmbreeRaycaster.cpp
)
endif(embree_FOUND)

if(WITH_FREENECT AND LIBFREENECT_FOUND)
set(LVR2_SOURCES ${LVR2_SOURCES} 
  io/KinectGrabber.cpp 
  io/KinectIO.cpp)
endif()

#####################################################################################
# Draco Geometry Compression
#####################################################################################

if(draco_FOUND)
set(LVR2_SOURCES ${LVR2_SOURCES}
  io/DracoEncoder.cpp
  io/DracoDecoder.cpp 
  io/DrcIO.cpp)
endif(draco_FOUND)

#####################################################################################
# RiVLib
#####################################################################################

if(RiVLib_FOUND)
  set(LVR2_SOURCES ${LVR2_SOURCES} 
    io/RxpIO.cpp)
endif()

#####################################################################################
# OpenCL
#####################################################################################

if(OPENCL_FOUND)
    list(APPEND LVR2_SOURCES
        reconstruction/opencl/ClSurface.cpp
        reconstruction/opencl/ClStatisticalOutlierFilter.cpp)
endif()

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

#####################################################################################
# Set c++0x flags for gcc compilers (needed for boctree io)
#####################################################################################

if(UNIX)
  SET_SOURCE_FILES_PROPERTIES(io/BoctreeIO.cpp PROPERTIES COMPILE_FLAGS "-std=c++14")
endif(UNIX)


#####################################################################################
# Compile object files for static and dynamic library
#####################################################################################

add_library(lvr2core OBJECT ${LVR2_SOURCES})

#####################################################################################
# Build static library
#####################################################################################

message(STATUS "Building static library")
add_library(lvr2_static STATIC $<TARGET_OBJECTS:lvr2core>)
target_link_libraries(
  lvr2_static 
  ${LVR2_INTERNAL_DEPENDENCIES_STATIC} 
  ${LVR2_LIB_DEPENDENCIES})


#####################################################################################
# Build shared library
#####################################################################################

if( NOT MSVC )
  message(STATUS "Building shared library")
  add_library(lvr2 SHARED $<TARGET_OBJECTS:lvr2core>)
  target_link_libraries(
    lvr2 
    ${LVR2_INTERNAL_DEPENDENCIES_SHARED} 
    ${LVR2_LIB_DEPENDENCIES})

install(TARGETS lvr2_static lvr2
  EXPORT lvr2Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
endif( NOT MSVC)

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

