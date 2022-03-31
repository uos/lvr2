set(LVR2_SOURCES
    algorithm/ChunkBuilder.cpp
    algorithm/ChunkManager.cpp
    algorithm/ChunkHashGrid.cpp
    algorithm/pmp/DifferentialGeometry.cpp
    algorithm/pmp/DistancePointTriangle.cpp
    algorithm/pmp/DifferentialGeometry.cpp
    algorithm/pmp/SurfaceFactory.cpp
    algorithm/pmp/SurfaceGeodesic.cpp
    algorithm/pmp/SurfaceParameterization.cpp
    algorithm/pmp/SurfaceSmoothing.cpp
    algorithm/pmp/TriangleKdTree.cpp
    algorithm/pmp/DistancePointTriangle.cpp
    algorithm/pmp/SurfaceFairing.cpp
    algorithm/pmp/SurfaceHoleFilling.cpp
    algorithm/pmp/SurfaceRemeshing.cpp
    algorithm/pmp/SurfaceSubdivision.cpp
    algorithm/pmp/SurfaceCurvature.cpp
    algorithm/pmp/SurfaceFeatures.cpp
    algorithm/pmp/SurfaceNormals.cpp
    algorithm/pmp/SurfaceSimplification.cpp
    algorithm/pmp/SurfaceTriangulation.cpp
    config/lvropenmp.cpp
    config/BaseOption.cpp
    display/Renderable.cpp
    # display/GroundPlane.cpp
    display/MultiPointCloud.cpp
    display/StaticMesh.cpp
    display/Color.cpp
    display/Grid.cpp
    display/PointCloud.cpp
    display/InteractivePointCloud.cpp
    display/CoordinateAxes.cpp
    display/GlTexture.cpp
    display/PointCorrespondences.cpp
    # display/Arrow.cpp
    display/TexturedMesh.cpp
    display/MeshCluster.cpp
    geometry/pmp/SurfaceMesh.cpp
    geometry/pmp/SurfaceMeshIO.cpp
    geometry/SoilAssistField.cpp
    geometry/SoilAssistSubField.cpp
    io/baseio/yaml/Matrix.cpp
    io/modelio/AsciiIO.cpp
    io/modelio/ModelIOBase.cpp
    io/modelio/ObjIO.cpp
    io/modelio/DatIO.cpp
    io/modelio/LasIO.cpp
    io/modelio/GeoTIFFIO.cpp
#    io/KinectIO.cpp
    io/AttributeMeshIOBase.cpp
    io/modelio/PPMIO.cpp
    io/modelio/PLYIO.cpp
    io/modelio/STLIO.cpp
    io/modelio/UosIO.cpp
    io/modelio/PCDIO.cpp
    io/kernels/DirectoryKernel.cpp
    io/kernels/HDF5Kernel.cpp
    io/LineReader.cpp
    # io/HDF5IO.cpp
    io/GridIO.cpp
    io/ModelFactory.cpp
    # io/ScanDataManager.cpp
    io/ScanDirectoryParser.cpp
    io/schema/ScanProjectSchemaRaw.cpp
    io/schema/ScanProjectSchemaSlam6D.cpp
    io/schema/ScanProjectSchemaEuRoC.cpp
    io/schema/ScanProjectSchema.cpp
    io/schema/MeshSchemaDirectory.cpp
    io/schema/MeshSchemaHDF5.cpp
    # io/schema/ScanProjectSchemaSLAM.cpp
    # io/schema/ScanProjectSchemaHyperlib.cpp
    io/schema/ScanProjectSchemaHDF5.cpp
    # io/scanio/LabelScanProjectSchemaHDF5V2.cpp
    io/scanio/DirectoryDataIO.cpp
    io/scanio/MetaFormatFactory.cpp
    # io/scanio/HDF5MetaDescriptionV2.cpp
    reconstruction/Projection.cpp
    reconstruction/PanoramaNormals.cpp
    reconstruction/ModelToImage.cpp
    reconstruction/LBKdTree.cpp
    registration/ICPPointAlign.cpp
    registration/KDTree.cpp
    registration/SLAMScanWrapper.cpp
    registration/Metascan.cpp
    registration/SLAMAlign.cpp
    registration/GraphSLAM.cpp
    registration/OctreeReduction.cpp
    registration/RegistrationPipeline.cpp
    types/CustomChannelTypes.cpp
    types/MeshBuffer.cpp
    types/PolygonBuffer.cpp
    types/PointBuffer.cpp
    texture/Texture.cpp
    texture/TextureFactory.cpp
    util/ColorGradient.cpp
    util/CoordinateTransform.cpp
    util/Hdf5Util.cpp
    util/IOUtils.cpp
    util/Synthetic.cpp
    util/TreeUtils.cpp
    util/Util.cpp
    util/Progress.cpp
    util/Timestamp.cpp
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
  io/baseio/DracoEncoder.cpp
  io/baseio/DracoDecoder.cpp 
  io/baseio/DrcIO.cpp)
endif(draco_FOUND)

#####################################################################################
# RiVLib
#####################################################################################

if(RiVLib_FOUND)
  set(LVR2_SOURCES ${LVR2_SOURCES} 
    io/baseio/RxpIO.cpp)
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
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_INTERNAL_DEPENDENCIES_STATIC
    lvr2rply_static
    lvr2las_static)

set(LVR2_INTERNAL_DEPENDENCIES_SHARED
    lvr2rply
    lvr2las)

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

