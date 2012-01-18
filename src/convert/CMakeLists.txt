
message( STATUS "Creating target convert…" )
add_executable( convertmodel convert.cpp )
target_link_libraries( convertmodel staticlssr )

if(PCL_FOUND)
    target_link_libraries( convertmodel ${PCL_LIBRARIES} )
endif(PCL_FOUND)
