# External dependencies to ANN, NEWMAT and STANN

set(MCUBES_SOURCES 
FastBox.cpp 
FastGrid.cpp 
Options.cpp 
Progress.cpp
QueryPoint.cpp 
StannInterpolator.cpp
Tetraeder.cpp 
TetraederBox.cpp
Timestamp.cpp)

add_executable(mcubes main.cc ${MCUBES_SOURCES})

set_target_properties(mcubes PROPERTIES BINARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

target_link_libraries(mcubes boost_system boost_program_options boost_filesystem boost_thread)
target_link_libraries(mcubes model3d)
target_link_libraries(mcubes ${OPENGL_LIBRARY})
target_link_libraries(mcubes ${OpenMP_CXX_FLAGS})
