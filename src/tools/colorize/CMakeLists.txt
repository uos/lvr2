add_executable( lvr_colorize colorize.cpp )
target_link_libraries( lvr_colorize lvr_static )

if(PCL_FOUND)
    target_link_libraries( lvr_colorize ${PCL_LIBRARIES} )
endif(PCL_FOUND)
if( ${NABO_FOUND} )
    target_link_libraries( lvr_colorize ${NABO_LIBRARY})
endif( ${NABO_FOUND} )

install(TARGETS lvr_colorize
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
