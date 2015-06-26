set(CMAKE_CXX_COMPILER "mpic++")

add_executable(lvr_largescale_reconstruct Main.cpp)
target_link_libraries(lvr_largescale_reconstruct lvr_static)
