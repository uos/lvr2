set(LSSR_SOURCES
	io/ModelFactory.cpp
    io/PLYIO.cpp
    io/AsciiIO.cpp
    io/UosIO.cpp
    io/Progress.cpp
    io/Timestamp.cpp
    io/Message.cpp
    io/MeshBuffer.cpp
    io/PointBuffer.cpp
    display/CoordinateAxes.cpp
    display/Renderable.cpp
    display/PointCloud.cpp
    display/MultiPointCloud.cpp
    display/StaticMesh.cpp
    display/ColorMap.cpp
    display/GroundPlane.cpp
)


add_library(lssr STATIC ${LSSR_SOURCES})

set_target_properties(lssr PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
target_link_libraries(lssr ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(lssr ${OPENGL_LIBRARY})
if ( rply_shared ) 
	target_link_libraries(lssr ${rply_shared})
else ( rply_shared )
	target_link_libraries(lssr rply_local_static)
endif ( rply_shared )
target_link_libraries(lssr ${OpenMP_CXX_FLAGS})
