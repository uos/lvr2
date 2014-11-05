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

set(LVR_ASCIICONVERT_DEPENDENCIES 
	lvr_static
	lvrlas_static
	lvrrply_static
	lvrslam6d_static
	${OPENGL_LIBRARY} 
	${GLUT_LIBRARIES}
	)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr_ascii_convert ${ASCII_SOURCES})
set_target_properties(lvr_ascii_convert PROPERTIES BINARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(lvr_ascii_convert ${LVR_ASCIICONVERT_DEPENDENCIES})

