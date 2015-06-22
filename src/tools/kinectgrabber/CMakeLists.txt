set(GRABBER_SOURCES   
    Main.cpp
    Options.cpp
)

add_executable(lvr_kinect_grabber ${GRABBER_SOURCES})

target_link_libraries(lvr_kinect_grabber staticlvr)
target_link_libraries(lvr_kinect_grabber ${Boost_FILESYSTEM_LIBRARY}
                              ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
                              ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(lvr_kinect_grabber ${OpenMP_CXX_FLAGS})
target_link_libraries(lvr_kinect_grabber ${PCL_LIBRARIES})
target_link_libraries(lvr_kinect_grabber freenect)
