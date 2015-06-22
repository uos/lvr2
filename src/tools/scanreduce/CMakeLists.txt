#####################################################################################
# Set source files
#####################################################################################

set(LVR_REDUCE_SOURCES
    Options.cpp     
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries 
#####################################################################################

set(LVR_REDUCE_DEPENDENCIES 
	lvr_static
	lvrlas_static
	lvrrply_static
	lvrslam6d_static
	${OPENGL_LIBRARY} 
	${GLUT_LIBRARIES}
	${OpenCV_LIBS}
	)

if(PCL_FOUND)
   set(LVR_REDUCE_DEPENDENCIES  ${LVR_REDUCE_DEPENDENCIES} ${PCL_LIBRARIES} )
endif(PCL_FOUND)

if( ${NABO_FOUND} )
   set(LVR_REDUCE_DEPENDENCIES  ${LVR_REDUCE_DEPENDENCIES} ${NABO_LIBRARY} )
endif( ${NABO_FOUND} )


#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr_pointcloud_reduce ${LVR_REDUCE_SOURCES})
target_link_libraries(lvr_pointcloud_reduce ${LVR_REDUCE_DEPENDENCIES})
