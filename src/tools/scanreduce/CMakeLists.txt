
set(RECONSTRUCT_SOURCES
    Options.cpp     
    Main.cpp
)

add_executable(lvr_pointcloud_reduce ${RECONSTRUCT_SOURCES})
set_target_properties(lvr_pointcloud_reduce PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(lvr_pointcloud_reduce staticlvr)
target_link_libraries(lvr_pointcloud_reduce ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})

INSTALL(TARGETS lvr_pointcloud_reduce
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
