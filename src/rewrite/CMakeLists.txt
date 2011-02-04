#External dependencies to ANN, NEWMAT and STANN

set(LSSR_SOURCES
 
)

add_executable(lssr Main.cpp ${LSSR_SOURCES})

set_target_properties(lssr PROPERTIES BINARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

target_link_libraries(lssr boost_system boost_program_options boost_filesystem boost_thread)
target_link_libraries(lssr model3d)
target_link_libraries(lssr ann)
target_link_libraries(lssr ${OPENGL_LIBRARY})
target_link_libraries(lssr ${OpenMP_CXX_FLAGS})

