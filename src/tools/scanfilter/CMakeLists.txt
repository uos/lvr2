
set(FILTER_SOURCES
    Options.cpp     
    Main.cpp
)

add_executable(lvr_pointcloud_filter ${FILTER_SOURCES})

target_link_libraries(lvr_pointcloud_filter lvr_static)
target_link_libraries(lvr_pointcloud_filter ${GLUT_LIBRARIES})

target_link_libraries(lvr_pointcloud_filter ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
    ${Boost_PROGRAM_OPTIONS_LIBRARY})
    
target_link_libraries(lvr_pointcloud_filter ${PCL_LIBRARIES})

