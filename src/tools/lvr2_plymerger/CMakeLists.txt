#####################################################################################
# Set source files
#####################################################################################

set(PLY_MERGE_SRC
    Options.cpp
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR_PLY_MERGER_DEPENDENCIES
	lvr2_static
	lvrlas_static
	lvrrply_static
	lvrslam6d_static
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_ply_merger ${PLY_MERGE_SRC})
target_link_libraries(lvr2_ply_merger ${LVR_PLY_MERGER_DEPENDENCIES})

install(TARGETS lvr2_ply_merger
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
