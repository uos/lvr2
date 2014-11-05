set(LVR_SOURCES
    io/BaseIO.cpp
    io/ModelFactory.cpp
    io/PLYIO.cpp
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
#    io/KinectGrabber.cpp
#    io/KinectIO.cpp
    io/CoordinateTransform.cpp
    io/BoctreeIO.cpp
    io/TextureIO.cpp
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
    reconstruction/PCLFiltering.cpp
    registration/EigenSVDPointAlign.cpp
    registration/ICPPointAlign.cpp
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
# Setup dependencies to external libraries 
#####################################################################################

set(LVR_LIB_DEPENDENCIES 
	lvrrply_static
	lvrlas_static
	lvrslam6d_static 
	${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} 
	${Boost_THREAD_LIBRARY} 
	${Boost_FILESYSTEM_LIBRARY}
	${OPENGL_LIBRARY} 
	${GLUT_LIBRARIES}
	${OpenCV_LIBS})
    	
#####################################################################################
# On Unix/Linux we need to link against pthreads 
#####################################################################################

if ( UNIX )
set(LVR_LIB_DEPENDENCIES $(LVR_LIB_DEPENDENCIES) pthread)
endif ( UNIX)

#####################################################################################
# Add QT dependencies if found
#####################################################################################

if( QT4_FOUND )
set(LVR_LIB_DEPENDENCIES $(LVR_LIB_DEPENDENCIES) 
	${QT_QTGUI_LIBRARIES} 
	${QT_QTCORE_LIBRARIES})
endif (QT4_FOUND)

#####################################################################################
# Set c++0x flags for gcc compilers (needed for boctree io)
#####################################################################################

if( UNIX )
SET_SOURCE_FILES_PROPERTIES(io/BoctreeIO.cpp PROPERTIES COMPILE_FLAGS "-std=gnu++0x")
endif ( UNIX )

#####################################################################################
# Add PCD io if PCL is installed
#####################################################################################

if(PCL_FOUND)
  set(LVR_SOURCES ${LVR_SOURCES} io/PCDIO.cpp)
endif(PCL_FOUND)

#####################################################################################
# Compile object files for static and dynamic library
#####################################################################################

add_library( lvrcore OBJECT ${LVR_SOURCES})

#####################################################################################
# Build static library
#####################################################################################

message(STATUS "Building static library")
add_library(lvr_static STATIC $<TARGET_OBJECTS:lvrcore>)
set_target_properties(lvr_static PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
target_link_libraries(lvr_static ${LVR_LIB_DEPENDENCIES})


#####################################################################################
# Build shared library
#####################################################################################

message(STATUS "Building shared library")
add_library(lvr SHARED $<TARGET_OBJECTS:lvrcore>)
set_target_properties(lvr PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
#target_link_libraries(lvr ${LVR_LIB_DEPENDENCIES})




