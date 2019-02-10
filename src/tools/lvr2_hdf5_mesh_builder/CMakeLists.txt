#####################################################################################
# Set source files
#####################################################################################

set(LVR_HDF5_MESH_SOURCES
    HDF5MeshTool.cpp
    Options.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR_HDF5_MESH_DEPENDENCIES
	lvr2_static
	lvrlas_static
	lvrrply_static
	lvrslam6d_static
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
)


#####################################################################################
# Add executable
#####################################################################################

add_executable( lvr2_hdf5_mesh_tool ${LVR_HDF5_MESH_SOURCES} )
target_link_libraries( lvr2_hdf5_mesh_tool ${LVR_HDF5_MESH_DEPENDENCIES} )

install(TARGETS lvr2_hdf5_mesh_tool
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
