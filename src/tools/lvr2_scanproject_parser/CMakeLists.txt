#####################################################################################
# Set source files
#####################################################################################

set(LVR2_SCANPROJECT_PARSER_SOURCES
    Main.cpp
    Options.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_SCANPROJECT_PARSER_DEPENDENCIES
	lvr2_static
	lvr2las_static
	lvr2rply_static
	lvr2slam6d_static
	${LVR2_LIB_DEPENDENCIES}
)


#####################################################################################
# Add executable
#####################################################################################

add_executable( lvr2_scanproject_parser ${LVR2_SCANPROJECT_PARSER_SOURCES} )
target_link_libraries( lvr2_scanproject_parser ${LVR2_SCANPROJECT_PARSER_DEPENDENCIES} )

install(TARGETS lvr2_scanproject_parser
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
