

set(CMAKE_CXX_COMPILER "mpic++")


add_executable(MPI ownmpi.cpp)
target_link_libraries(MPI_test staticlssr)

