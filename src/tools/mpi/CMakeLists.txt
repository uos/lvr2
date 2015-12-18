

add_executable(lvr_mpi_reconstruct Main.cpp )
add_executable(seg LargeScaleReconstruction.cpp NodeData.cpp LargeScaleOctree.cpp)
target_link_libraries(lvr_mpi_reconstruct lvr_static)
target_link_libraries(seg  ${Boost_LIBRARIES} ${MPI_LIBRARIES} lvr_static )
install(TARGETS lvr_mpi_reconstruct
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
