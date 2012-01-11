
message( STATUS "Creating target colorize…" )
add_executable( colorize colorize.cpp )
target_link_libraries( colorize staticlssr )

if(PCL_FOUND)
    target_link_libraries( colorize ${PCL_LIBRARIES} )
endif(PCL_FOUND)
