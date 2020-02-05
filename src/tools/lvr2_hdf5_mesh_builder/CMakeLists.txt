#####################################################################################
# Set source files
#####################################################################################

set(LVR2_HDF5_MESH_SOURCES
    HDF5MeshTool.cpp
    Options.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_HDF5_MESH_DEPENDENCIES
	lvr2_static
	lvr2las_static
	lvr2rply_static
	lvr2slam6d_static
	${LVR2_LIB_DEPENDENCIES}
)


#####################################################################################
# Add executable
#####################################################################################

add_executable( lvr2_hdf5_mesh_tool ${LVR2_HDF5_MESH_SOURCES} )
target_link_libraries( lvr2_hdf5_mesh_tool ${LVR2_HDF5_MESH_DEPENDENCIES} )

install(TARGETS lvr2_hdf5_mesh_tool
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
