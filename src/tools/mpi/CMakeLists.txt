
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
add_executable(seg LargeScaleReconstruction.cpp NodeData.cpp LargeScaleOctree.cpp Options.cpp)
target_link_libraries(seg  ${Boost_LIBRARIES} "/opt/sgi/mpt/mpt-2.08/lib/libmpi.so" "/opt/sgi/mpt/mpt-2.08/lib/libmpi++.so" lvr_static )
