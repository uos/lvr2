set(CMAKE_CXX_COMPILER "mpic++")

add_executable(lvr_mpi_reconstruct Main.cpp )
add_executable(seg segmentize.cpp NodeData.cpp NodeData.h)
target_link_libraries(lvr_mpi_reconstruct lvr_static)

install(TARGETS lvr_mpi_reconstruct
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
