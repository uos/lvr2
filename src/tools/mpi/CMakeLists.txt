
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
add_executable(seg LargeScaleReconstruction.cpp NodeData.cpp LargeScaleOctree.cpp)
target_link_libraries(seg  ${Boost_LIBRARIES} ${MPI_CXX_LIBRARIES} lvr_static )
