set(LSSR_SOURCES
    io/BaseIO.cpp
    io/ModelFactory.cpp
    io/PLYIO.cpp
    io/AsciiIO.cpp
    io/UosIO.cpp
    io/ObjIO.cpp	
    io/LasIO.cpp
    io/PCDIO.cpp
    io/PPMIO.cpp
    io/Progress.cpp
    io/Timestamp.cpp
    io/MeshBuffer.cpp
    io/PointBuffer.cpp
    io/GridIO.cpp
    io/KinectGrabber.cpp
    io/KinectIO.cpp
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
    texture/Texture.cpp
    texture/ImageProcessor.cpp
    texture/Statistics.cpp
    texture/AutoCorr.cpp
    texture/CrossCorr.cpp
    texture/CCV.cpp
    texture/Transform.cpp
    texture/Trans.cpp
)

message(STATUS "Building shared library")
add_library(lssr SHARED ${LSSR_SOURCES})

message(STATUS "Building static library")
add_library(staticlssr STATIC ${LSSR_SOURCES})

set_target_properties(staticlssr PROPERTIES LIBRARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/lib)

target_link_libraries(staticlssr ${Boost_PROGRAM_OPTIONS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_FILESYSTEM_LIBRARY}
    ${OpenMP_CXX_FLAGS} ${OPENGL_LIBRARY} ${OpenCV_LIBS})
target_link_libraries(staticlssr laslib_static)
target_link_libraries(staticlssr obj)
target_link_libraries(staticlssr pthread)
target_link_libraries(staticlssr ${PCL_LIBRARIES})
target_link_libraries(staticlssr slam6d_static)

if ( rply_shared ) 
    target_link_libraries(staticlssr ${rply_shared})
else ( rply_shared )
    target_link_libraries(staticlssr rply_local_static)
endif ( rply_shared )

if ( "${CMAKE_BUILD_TYPE}" STREQUAL "Release" ) 

    set_target_properties(lssr PROPERTIES LIBRARY_OUTPUT_DIRECTORY
         ${CMAKE_SOURCE_DIR}/lib)
    target_link_libraries(lssr ${Boost_PROGRAM_OPTIONS_LIBRARY}
         ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_FILESYSTEM_LIBRARY}
         ${OpenMP_CXX_FLAGS} ${OPENGL_LIBRARY} ${OpenCV_LIBS} )
    target_link_libraries(lssr laslib)
    if ( rply_shared ) 
         target_link_libraries(lssr ${rply_shared})
         target_link_libraries(staticlssr ${rply_shared})
    else ( rply_shared )
         target_link_libraries(lssr rply_local)
         target_link_libraries(staticlssr rply_local_static)
    endif ( rply_shared )
endif ( "${CMAKE_BUILD_TYPE}" STREQUAL "Release" ) 
