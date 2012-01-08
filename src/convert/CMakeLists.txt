
message( STATUS "Creating target convert…" )
add_executable( convertmodel convert.cpp )
target_link_libraries( convertmodel staticlssr )
