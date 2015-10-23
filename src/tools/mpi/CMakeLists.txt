set(CMAKE_CXX_COMPILER "mpic++")

add_executable(lvr_mpi_reconstruct Main.cpp)
target_link_libraries(lvr_mpi_reconstruct lvr_static)

install(TARGETS lvr_mpi_reconstruct
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
