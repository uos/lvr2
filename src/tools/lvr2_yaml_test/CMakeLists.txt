#####################################################################################
# Set source files
#####################################################################################

set(YAML_TEST_SOURCES
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_YAML_TEST_DEPENDENCIES
	lvr2_static
	lvr2las_static
	lvr2rply_static
	lvr2slam6d_static
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	${OpenCV_LIBS}
)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_yaml_test ${YAML_TEST_SOURCES})
target_link_libraries(lvr2_yaml_test ${LVR2_YAML_TEST_DEPENDENCIES})

install(TARGETS lvr2_yaml_test
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
