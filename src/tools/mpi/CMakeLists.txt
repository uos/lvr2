set(CMAKE_CXX_COMPILER "mpic++")

add_executable(lvr_mpi_reconstruct Main.cpp)
target_link_libraries(lvr_mpi_reconstruct staticlvr)

INSTALL(TARGETS lvr_mpi_reconstruct
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)