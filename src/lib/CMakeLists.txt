set(LSSR_SOURCES
	io/AsciiIO.cpp
	io/IOFactory.cpp
	io/MeshLoader.cpp
	io/PLYIO.cpp
	io/PointLoader.cpp
	io/Progress.cpp
	io/Timestamp.cpp
	io/UosIO.cpp
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
