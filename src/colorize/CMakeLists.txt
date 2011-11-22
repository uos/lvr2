
if(PCL_FOUND)

    message( STATUS "Creating target colorize…" )
    add_executable( colorize colorize.cpp )
    target_link_libraries( colorize lssr )
    target_link_libraries( colorize ${PCL_LIBRARIES} )

endif(PCL_FOUND)
