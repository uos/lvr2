set(SLICER_SOURCES  
    Options.cpp
    Main.cpp
)

add_executable(slicer ${SLICER_SOURCES})

set_target_properties(slicer PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)

target_link_libraries(slicer staticlssr)
target_link_libraries(slicer ${Boost_FILESYSTEM_LIBRARY}
							${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
							${Boost_PROGRAM_OPTIONS_LIBRARY})

target_link_libraries(slicer ${CGAL_LIBRARY})
