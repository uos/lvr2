
#####################################################################################
# Set source files
#####################################################################################

set(LVR_TIOGEN_SOURCES
   tiogen.cpp
)

#####################################################################################
# Setup dependencies to external libraries 
#####################################################################################

set(LVR_TIOGEN_DEPENDENCIES 
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
set(LVR_TIOGEN_DEPENDENCIES ${LVR_TIOGEN_DEPENDENCIES} ${NABO_LIBRARIES})
endif( ${NABO_FOUND} )

#####################################################################################
# Add executable
#####################################################################################

add_executable( lvr_tiogen ${LVR_TIOGEN_SOURCES} )
set_target_properties(lvr_tiogen PROPERTIES BINARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
TARGET_LINK_LIBRARIES( lvr_tiogen ${LVR_TIOGEN_DEPENDENCIES} )