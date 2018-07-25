#####################################################################################
# Set source files
#####################################################################################

set(LVR_HDF5_SOURCES
    HDF5Tool.cpp Options.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR_HDF5_DEPENDENCIES
	lvr_static
	lvrlas_static
	lvrrply_static
	lvrslam6d_static
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	)


#####################################################################################
# Add executable
#####################################################################################

add_executable( lvr_hdf5tool ${LVR_HDF5_SOURCES} )
target_link_libraries( lvr_hdf5tool ${LVR_HDF5_DEPENDENCIES} )

install(TARGETS lvr_hdf5tool
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
