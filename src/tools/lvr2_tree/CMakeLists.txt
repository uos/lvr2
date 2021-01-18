#####################################################################################
# Set source files
#####################################################################################

set(LVR2_TREE_SOURCES
	Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_TREE_DEPENDENCIES
	lvr2_static
	lvr2las_static
	lvr2rply_static
	lvr2slam6d_static
	${LVR2_LIB_DEPENDENCIES}
)


#####################################################################################
# Add executable
#####################################################################################

add_executable( lvr2_tree ${LVR2_TREE_SOURCES} )
target_link_libraries( lvr2_tree ${LVR2_TREE_DEPENDENCIES} )

install(TARGETS lvr2_tree
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
