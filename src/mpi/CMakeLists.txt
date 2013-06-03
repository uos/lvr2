

set(CMAKE_CXX_COMPILER "mpic++")


add_executable(MPI_test ownmpi.cpp)
target_link_libraries(MPI_test staticlssr)

add_executable( Normal_test tmpmain.cpp )
target_link_libraries(Normal_test staticlssr)
