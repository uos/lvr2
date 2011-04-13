#External dependencies to ANN, NEWMAT and STANN

set(LSSR_SOURCES
    lssr/Options.cpp     
    lssr/Timestamp.cpp   
    lssr/Progress.cpp
    io/PLYIO.cpp
    io/PLYTraits.cpp
)

add_library(lssr SHARED ${LSSR_SOURCES})
set_target_properties(lssr PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
target_link_libraries(lssr boost_system boost_program_options boost_filesystem boost_thread)
target_link_libraries(lssr model3d)
target_link_libraries(lssr ${OPENGL_LIBRARY})
target_link_libraries(lssr ${OpenMP_CXX_FLAGS})


add_executable(reconstruct lssr/Main.cpp)
set_target_properties(reconstruct PROPERTIES BINARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(reconstruct lssr)
target_link_libraries(reconstruct boost_system boost_program_options boost_filesystem boost_thread)
target_link_libraries(reconstruct model3d)
target_link_libraries(reconstruct ${OPENGL_LIBRARY})
target_link_libraries(reconstruct ${OpenMP_CXX_FLAGS})

