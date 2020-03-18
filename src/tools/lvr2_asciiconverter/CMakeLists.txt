#####################################################################################
# Set source files
#####################################################################################

set(ASCII_SOURCES
    Options.cpp
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_ASCIICONVERT_DEPENDENCIES
	lvr2_static
	lvr2las_static
	lvr2rply_static
	lvr2slam6d_static
	${LVR2_LIB_DEPENDENCIES}
)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr2_ascii_convert ${ASCII_SOURCES})
target_link_libraries(lvr2_ascii_convert ${LVR2_ASCIICONVERT_DEPENDENCIES})

install(TARGETS lvr2_ascii_convert
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
