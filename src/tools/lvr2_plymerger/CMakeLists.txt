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

set(LVR2_PLY_MERGER_DEPENDENCIES
	lvr2_static
	lvr2las_static
	lvr2rply_static
	lvr2slam6d_static
	${LVR2_LIB_DEPENDENCIES}
)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_ply_merger ${PLY_MERGE_SRC})
target_link_libraries(lvr2_ply_merger ${LVR2_PLY_MERGER_DEPENDENCIES})

install(TARGETS lvr2_ply_merger
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
