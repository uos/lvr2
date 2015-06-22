#####################################################################################
# Set source files
#####################################################################################

set(LEICA_SOURCES
    Options.cpp     
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries 
#####################################################################################

set(LVR_LEICA_DEPENDENCIES 
	lvr_static
	lvrlas_static
	lvrrply_static
	lvrslam6d_static
	${OPENGL_LIBRARY} 
	${GLUT_LIBRARIES}
	${OpenCV_LIBS}
	)
	
if(PCL_FOUND)
    set(LVR_LEICA_DEPENDENCIES ${LVR_LEICA_DEPENDENCIES} ${PCL_LIBRARIES} )
endif(PCL_FOUND)
	
#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr_leica_convert ${LEICA_SOURCES})
set_target_properties(lvr_leica_convert PROPERTIES BINARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(lvr_leica_convert ${LVR_LEICA_DEPENDENCIES})
