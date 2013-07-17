set(CMAKE_CXX_COMPILER "mpic++")

add_executable(mpi_reconstruct Main.cpp)
target_link_libraries(mpi_reconstruct staticlssr)

