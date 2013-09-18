set(FUSION_SOURCES  
    Options.cpp
    Main.cpp
)

add_executable(fusion ${FUSION_SOURCES})

set_target_properties(fusion PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)

target_link_libraries(fusion staticlvr)
target_link_libraries(fusion ${Boost_FILESYSTEM_LIBRARY}
							 ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
							 ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(fusion ${CGAL_LIBRARY})
