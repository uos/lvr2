
message( STATUS "Creating target texman…" )
add_executable( texman texman.cpp )
target_link_libraries( texman staticlssr )
