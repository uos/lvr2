add_executable( lvr_convertmodel Convert.cpp Options.cpp )
target_link_libraries( lvr_convertmodel staticlvr )

if(PCL_FOUND)
    target_link_libraries( lvr_convertmodel ${PCL_LIBRARIES} )
endif(PCL_FOUND)
