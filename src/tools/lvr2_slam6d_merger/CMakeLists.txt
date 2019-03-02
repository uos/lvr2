#####################################################################################
# Set source files
#####################################################################################

set(LVR_SLAM_MERGE_SOURCES
    Options.cpp
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR_SLAM_MERGE_DEPENDENCIES
	lvr2_static
	lvr2las_static
	lvr2rply_static
	lvr2slam6d_static
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	${OpenCV_LIBS}
)

if(PCL_FOUND)
    set(LVR_SLAM_MERGE_DEPENDENCIES ${PCL_LIBRARIES} )
endif(PCL_FOUND)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_slam6d_merger ${LVR_SLAM_MERGE_SOURCES})
target_link_libraries(lvr2_slam6d_merger ${LVR_SLAM_MERGE_DEPENDENCIES})

install(TARGETS lvr2_slam6d_merger
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
