#####################################################################################
# Set source files
#####################################################################################

set(CGNORMALS_SOURCES   
    Main.cpp
)

#####################################################################################
# Setup dependencies to external libraries 
#####################################################################################

set(CGAL_NORMAL_DEPENDENCIES 
	lvr_static
	lvrlas_static
	lvrrply_static
	lvrslam6d_static
	${OPENGL_LIBRARY} 
	${GLUT_LIBRARIES}
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	${CGAL_LIBRARY}
	)

#####################################################################################
# Add executable
#####################################################################################

add_executable(lvr_cgal_normal_estimation ${CGNORMALS_SOURCES})
set_target_properties(lvr_cgal_normal_estimation PROPERTIES BINARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(lvr_cgal_normal_estimation ${CGAL_NORMAL_DEPENDENCIES})


