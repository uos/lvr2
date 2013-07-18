set(CGNORMALS_SOURCES   
    Main.cpp
)

add_executable(cgal_normals ${CGNORMALS_SOURCES})

set_target_properties(cgal_normals PROPERTIES BINARY_OUTPUT_DIRECTORY
                              ${CMAKE_SOURCE_DIR}/bin)
    
target_link_libraries(cgal_normals staticlvr)
target_link_libraries(cgal_normals ${Boost_FILESYSTEM_LIBRARY}
                                    ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY}
                                    ${Boost_PROGRAM_OPTIONS_LIBRARY})
target_link_libraries(cgal_normals ${OpenMP_CXX_FLAGS})
target_link_libraries(cgal_normals ${PCL_LIBRARIES})
target_link_libraries(cgal_normals freenect)

target_link_libraries(cgal_normals ${CGAL_LIBRARY})