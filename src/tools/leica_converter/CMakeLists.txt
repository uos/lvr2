
set(LEICA_SOURCES
    Options.cpp     
    Main.cpp
)

add_executable(leica_convert ${LEICA_SOURCES})

set_target_properties(leica_convert PROPERTIES BINARY_OUTPUT_DIRECTORY
	${CMAKE_SOURCE_DIR}/bin)
	
target_link_libraries(leica_convert ${Boost_FILESYSTEM_LIBRARY}
	${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
	${Boost_PROGRAM_OPTIONS_LIBRARY})
	
target_link_libraries(leica_convert ${PCL_LIBRARIES})
target_link_libraries(leica_convert staticlvr)
