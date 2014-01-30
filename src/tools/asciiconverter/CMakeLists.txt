
set(ASCII_SOURCES
    Options.cpp     
    Main.cpp
)

add_executable(lvr_ascii_convert ${ASCII_SOURCES})

set_target_properties(lvr_ascii_convert PROPERTIES BINARY_OUTPUT_DIRECTORY
	${CMAKE_SOURCE_DIR}/bin)
	
target_link_libraries(lvr_ascii_convert ${Boost_FILESYSTEM_LIBRARY}
	${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
	${Boost_PROGRAM_OPTIONS_LIBRARY})
	
target_link_libraries(lvr_ascii_convert ${PCL_LIBRARIES})
target_link_libraries(lvr_ascii_convert staticlvr)
