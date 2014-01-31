set(SLICER_SOURCES  
    Options.cpp
    Main.cpp
)

add_executable(lvr_slicer ${SLICER_SOURCES})

set_target_properties(lvr_slicer PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)

target_link_libraries(lvr_slicer staticlvr)
target_link_libraries(lvr_slicer ${Boost_FILESYSTEM_LIBRARY}
							${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
							${Boost_PROGRAM_OPTIONS_LIBRARY})

target_link_libraries(lvr_slicer ${CGAL_LIBRARY})

INSTALL(TARGETS lvr_slicer
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
