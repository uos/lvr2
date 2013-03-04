set(GRABBER_SOURCES   
    Main.cpp
    Options.cpp
)

add_executable(kingrab ${GRABBER_SOURCES})

set_target_properties(kingrab PROPERTIES BINARY_OUTPUT_DIRECTORY
                              ${CMAKE_SOURCE_DIR}/bin)
    
target_link_libraries(kingrab staticlssr)
target_link_libraries(kingrab ${Boost_FILESYSTEM_LIBRARY}
                              ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
                              ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(kingrab ${OpenMP_CXX_FLAGS})
target_link_libraries(kingrab ${PCL_LIBRARIES})
target_link_libraries(kingrab freenect)