#####################################################################################
# Set source files
#####################################################################################

set(LVR2_SLAM_MERGE_SOURCES
    Options.cpp
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_SLAM_MERGE_DEPENDENCIES
	lvr2_static
	lvr2las_static
	lvr2rply_static
	lvr2slam6d_static
	${LVR2_LIB_DEPENDENCIES}
)

if(PCL_FOUND)
    list(APPEND LVR2_SLAM_MERGE_DEPENDENCIES ${PCL_LIBRARIES} )
endif(PCL_FOUND)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_slam6d_merger ${LVR2_SLAM_MERGE_SOURCES})
target_link_libraries(lvr2_slam6d_merger ${LVR2_SLAM_MERGE_DEPENDENCIES})

install(TARGETS lvr2_slam6d_merger
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
