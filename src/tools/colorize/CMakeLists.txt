
message( STATUS "Creating target colorize…" )
add_executable( lvr_colorize colorize.cpp )
target_link_libraries( lvr_colorize staticlvr )

if(PCL_FOUND)
    target_link_libraries( lvr_colorize ${PCL_LIBRARIES} )
endif(PCL_FOUND)
if( ${NABO_FOUND} )
    target_link_libraries( lvr_colorize ${NABO_LIBRARIES})
endif( ${NABO_FOUND} )
