#####################################################################################
# Set source files
#####################################################################################

set(LVR2_HDF5_SOURCES
    HDF5Tool.cpp
    Options.cpp
)

#####################################################################################
# Setup dependencies to external libraries
#####################################################################################

set(LVR2_HDF5_DEPENDENCIES
	lvr2_static
	lvr2las_static
	lvr2rply_static
	lvr2slam6d_static
	${LVR2_LIB_DEPENDENCIES}
)


#####################################################################################
# Add executable
#####################################################################################

add_executable( lvr2_hdf5tool ${LVR2_HDF5_SOURCES} )
target_link_libraries( lvr2_hdf5tool ${LVR2_HDF5_DEPENDENCIES} )

install(TARGETS lvr2_hdf5tool
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
