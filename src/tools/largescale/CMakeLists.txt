
FIND_PACKAGE(Boost 1.4 COMPONENTS mpi serialization REQUIRED)
find_package(MPI REQUIRED)

add_executable(lvr_largescale_reconstruct Main.cpp)

set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
include_directories(MPI_INCLUDE_PATH)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
target_link_libraries(lvr_largescale_reconstruct ${MPI_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(lvr_largescale_reconstruct lvr_static  )