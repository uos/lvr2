
if(PCL_FOUND)
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

    #    include_directories(${PCL_INCLUDE_DIRS})
    #    link_directories(${PCL_LIBRARY_DIRS})
    #    add_definitions(${PCL_DEFINITIONS})

    message( STATUS "Creating target colorize…" )
    add_executable( colorize colorize.cpp )
    target_link_libraries( colorize ${PCL_LIBRARIES} )

endif(PCL_FOUND)
