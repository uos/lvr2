
set(RECONSTRUCT_SOURCES
    Options.cpp     
    Main.cpp
)

add_executable(reconstruct ${RECONSTRUCT_SOURCES})
set_target_properties(reconstruct PROPERTIES BINARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(reconstruct lssr)
target_link_libraries(reconstruct boost_system boost_program_options boost_filesystem boost_thread)
target_link_libraries(reconstruct model3d)
target_link_libraries(reconstruct ${OPENGL_LIBRARY})
target_link_libraries(reconstruct ${OpenMP_CXX_FLAGS})

