

set(CMAKE_CXX_COMPILER "mpic++")


add_executable(MPI_test ownmpi.cpp)
target_link_libraries(MPI_test staticlssr)

#add_executable( kd_test mpi/tmpmain.cpp )
#target_link_libraries(kd_test staticlssr)
