#####################################################################################
# Set source files
#####################################################################################

set(LVR_TRANSFORM_SOURCES
    Options.cpp
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries 
#####################################################################################

set(LVR_TRANSFORM_DEPENDENCIES 
	lvr_static
	lvrlas_static
	lvrrply_static
	lvrslam6d_static
	${OPENGL_LIBRARY} 
	${GLUT_LIBRARIES}
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	)

if( ${NABO_FOUND} )
set(LVR_TRANSFORM_DEPENDENCIES ${LVR_TRANSFORM_DEPENDENCIES} ${NABO_LIBRARIES})
endif( ${NABO_FOUND} )


#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr_transform_model ${LVR_TRANSFORM_SOURCES})
target_link_libraries(lvr_transform_model ${LVR_TRANSFORM_DEPENDENCIES})
