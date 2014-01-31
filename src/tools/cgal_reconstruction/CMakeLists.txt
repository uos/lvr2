set(CGNORMALS_SOURCES   
    Main.cpp
)

add_executable(lvr_cgal_normal_estimation ${CGNORMALS_SOURCES})

set_target_properties(lvr_cgal_normal_estimation PROPERTIES BINARY_OUTPUT_DIRECTORY
                              ${CMAKE_SOURCE_DIR}/bin)
    
target_link_libraries(lvr_cgal_normal_estimation staticlvr)
target_link_libraries(lvr_cgal_normal_estimation ${Boost_FILESYSTEM_LIBRARY}
                                    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
                                    ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(lvr_cgal_normal_estimation ${OpenMP_CXX_FLAGS})
target_link_libraries(lvr_cgal_normal_estimation ${PCL_LIBRARIES})
target_link_libraries(lvr_cgal_normal_estimation freenect)

target_link_libraries(lvr_cgal_normal_estimation ${CGAL_LIBRARY})

INSTALL(TARGETS lvr_cgal_normal_estimation
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
