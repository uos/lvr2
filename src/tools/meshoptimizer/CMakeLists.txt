#####################################################################################
# Set source files
#####################################################################################

set(MESH_OPT_SRC
    Options.cpp     
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries 
#####################################################################################

set(LVR_OPT_DEPENDENCIES 
	lvr_static
	lvrlas_static
	lvrrply_static
	lvrslam6d_static
	${OPENGL_LIBRARY} 
	${GLUT_LIBRARIES}
	${OpenCV_LIBS}
	)

if(PCL_FOUND)
    set(LVR_OPT_DEPENDENCIES ${LVR_OPT_DEPENDENCIES} ${PCL_LIBRARIES} )
endif(PCL_FOUND)

if( ${NABO_FOUND} )
   set(LVR_OPT_DEPENDENCIES ${LVR_OPT_DEPENDENCIES} ${NABO_LIBRARY} )
endif( ${NABO_FOUND} )

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr_mesh_optimizer ${MESH_OPT_SRC})
set_target_properties(lvr_mesh_optimizer PROPERTIES BINARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(lvr_mesh_optimizer ${LVR_OPT_DEPENDENCIES} )
   
