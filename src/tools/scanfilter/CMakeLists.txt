
set(FILTER_SOURCES
    Options.cpp     
    Main.cpp
)

add_executable(lvr_pointcloud_filter ${FILTER_SOURCES})

set_target_properties(lvr_pointcloud_filter PROPERTIES BINARY_OUTPUT_DIRECTORY
    ${CMAKE_SOURCE_DIR}/bin)
    
target_link_libraries(lvr_pointcloud_filter lvr)
target_link_libraries(lvr_pointcloud_filter glut)
target_link_libraries(lvr_pointcloud_filter freenect)

target_link_libraries(lvr_pointcloud_filter ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
    
target_link_libraries(lvr_pointcloud_filter ${PCL_LIBRARIES})

INSTALL(TARGETS lvr_pointcloud_filter
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
