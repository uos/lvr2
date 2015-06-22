
#####################################################################################
# Set source files
#####################################################################################

set(LVR_TEXMAN_SOURCES
   texman.cpp
)

#####################################################################################
# Setup dependencies to external libraries 
#####################################################################################

set(LVR_TEXMAN_DEPENDENCIES 
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
set(LVR_TEXMAN_DEPENDENCIES ${LVR_TEXMAN_DEPENDENCIES} ${NABO_LIBRARIES})
endif( ${NABO_FOUND} )

#####################################################################################
# Add executable
#####################################################################################


add_executable( lvr_texman  ${LVR_TEXMAN_SOURCES})
TARGET_LINK_LIBRARIES( lvr_texman ${LVR_TEXMAN_DEPENDENCIES} )


