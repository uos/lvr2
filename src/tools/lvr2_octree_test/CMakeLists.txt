#####################################################################################
# Set source files
#####################################################################################

set(OCTREE_TEST_SOURCES
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_OCTREE_TEST_DEPENDENCIES
	lvr2_static
	lvr2las_static
	lvr2rply_static
	${LVR2_LIB_DEPENDENCIES}
)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_octree_test ${OCTREE_TEST_SOURCES})
target_link_libraries(lvr2_octree_test ${LVR2_OCTREE_TEST_DEPENDENCIES})
target_precompile_headers(lvr2_octree_test PUBLIC  ${LVR2_PCHS})
install(TARGETS lvr2_octree_test
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
