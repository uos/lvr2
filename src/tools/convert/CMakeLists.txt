#####################################################################################
# Set source files
#####################################################################################

set(LVR_CONVERT_SOURCES   
    Convert.cpp Options.cpp
)

#####################################################################################
# Setup dependencies to external libraries 
#####################################################################################

set(LVR_CONVERT_DEPENDENCIES 
	lvr_static
	lvrlas_static
	lvrrply_static
	lvrslam6d_static
	${OPENGL_LIBRARY} 
	${GLUT_LIBRARIES}
	)

if(PCL_FOUND)
    set(LVR_CONVERT_DEPENDENCIES ${LVR_CONVERT_DEPENDENCIES} ${PCL_LIBRARIES} )
endif(PCL_FOUND)

#####################################################################################
# Add executable
#####################################################################################

add_executable( lvr_convertmodel ${LVR_CONVERT_SOURCES} )
target_link_libraries( lvr_convertmodel ${LVR_CONVERT_DEPENDENCIES} )
