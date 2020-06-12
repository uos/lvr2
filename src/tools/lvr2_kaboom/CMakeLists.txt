#####################################################################################
# Set source files
#####################################################################################

set(KABOOM_SOURCES
    Options.cpp
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_KABOOM_DEPENDENCIES
	lvr2_static
	lvr2las_static
	lvr2rply_static
	lvr2slam6d_static
	${LVR2_LIB_DEPENDENCIES}
)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_kaboom ${KABOOM_SOURCES})
target_link_libraries(lvr2_kaboom ${LVR2_KABOOM_DEPENDENCIES})

install(TARGETS lvr2_kaboom
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
